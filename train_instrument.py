"""
train_instrument.py
===================
Train a neural instrument synthesizer for Indian classical instruments.

Architecture: RAVE-inspired VAE with harmonic + noise decoder
- Encoder: learns latent representation of instrument timbre from audio
- Decoder: harmonic synthesizer + filtered noise (like a real string/wind instrument)
- The model learns HOW the instrument sounds at every pitch and dynamic

Usage:
    py -3.11 train_instrument.py --instrument veena
    py -3.11 train_instrument.py --instrument bansuri
    py -3.11 train_instrument.py --instrument sitar

After training, the model is saved to models/<instrument>/
The audio engine picks it up automatically.

Training time on CPU:
    ~4-6 hours for full model with 30+ min audio
    Checkpoint saved every 50 epochs so you can stop and resume

Resume interrupted training:
    py -3.11 train_instrument.py --instrument veena --resume
"""

import argparse
import math
import os
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SR         = 44100
HOP_LENGTH = 256       # ~5.8ms per frame at 44100
N_FFT      = 1024
N_MELS     = 128
FRAME_SECS = 2.0       # training clip length
FRAME_LEN  = int(SR * FRAME_SECS)
LATENT_DIM = 64
N_HARMONICS = 64       # number of harmonic overtones to model
N_NOISE     = 65       # noise filter bands

PROJECT  = Path(__file__).parent
SAMPLES  = PROJECT / "samples"
MODELS   = PROJECT / "models"

# ---------------------------------------------------------------------------
# Dataset — loads sliced WAV samples + raw recordings
# ---------------------------------------------------------------------------

class InstrumentDataset(torch.utils.data.Dataset):
    """
    Loads all audio for an instrument:
    1. Sliced note WAVs from samples/<instrument>/
    2. Raw recordings from samples/raw/<instrument>/

    Each item is a FRAME_LEN audio chunk.
    """

    def __init__(self, instrument: str, augment: bool = True):
        self.augment = augment
        self.chunks  = []

        # Load sliced notes (loop each to fill FRAME_LEN)
        note_dir = SAMPLES / instrument
        if note_dir.exists():
            for wav in sorted(note_dir.glob("*.wav")):
                y = self._load(wav)
                if y is not None:
                    # Loop note to fill frame length
                    while len(y) < FRAME_LEN:
                        y = np.concatenate([y, y])
                    # Extract multiple chunks from each note
                    for start in range(0, len(y) - FRAME_LEN, FRAME_LEN // 2):
                        chunk = y[start:start + FRAME_LEN]
                        if len(chunk) == FRAME_LEN:
                            self.chunks.append(chunk.astype(np.float32))

        # Load raw recordings (first 5 min only)
        raw_dir = SAMPLES / "raw" / instrument
        if raw_dir.exists():
            exts = {".wav", ".mp3", ".ogg", ".flac"}
            for f in sorted(raw_dir.glob("*")):
                if f.suffix.lower() in exts:
                    y = self._load(f, max_secs=300)
                    if y is not None:
                        for start in range(0, len(y) - FRAME_LEN, FRAME_LEN):
                            chunk = y[start:start + FRAME_LEN]
                            if len(chunk) == FRAME_LEN:
                                self.chunks.append(chunk.astype(np.float32))

        print(f"  Dataset: {len(self.chunks)} chunks "
              f"({len(self.chunks) * FRAME_SECS / 60:.1f} min) for {instrument}")

    def _load(self, path: Path, max_secs: float = None) -> np.ndarray | None:
        try:
            y, sr = librosa.load(str(path), sr=SR, mono=True)
            if max_secs:
                y = y[:int(max_secs * SR)]
            return y.astype(np.float32)
        except Exception as e:
            print(f"  Could not load {path.name}: {e}")
            return None

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        y = self.chunks[idx].copy()

        if self.augment:
            # Random gain (-6dB to +3dB)
            gain = 10 ** (np.random.uniform(-6, 3) / 20)
            y    = y * gain

            # Random pitch shift ±2 semitones
            if np.random.random() < 0.3:
                steps = np.random.uniform(-2, 2)
                y = librosa.effects.pitch_shift(y, sr=SR, n_steps=steps)

            # Random time stretch (0.95 to 1.05)
            if np.random.random() < 0.2:
                rate = np.random.uniform(0.95, 1.05)
                y = librosa.effects.time_stretch(y, rate=rate)
                # Re-trim/pad to FRAME_LEN
                if len(y) > FRAME_LEN:
                    start = np.random.randint(0, len(y) - FRAME_LEN)
                    y = y[start:start + FRAME_LEN]
                else:
                    y = np.pad(y, (0, FRAME_LEN - len(y)))

        # Clip and normalize
        y = np.clip(y, -1.0, 1.0)
        return torch.FloatTensor(y)


# ---------------------------------------------------------------------------
# Feature extraction — mel spectrogram
# ---------------------------------------------------------------------------

class MelSpec(nn.Module):
    def __init__(self):
        super().__init__()
        # Pre-compute mel filterbank
        mel_fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
        self.register_buffer("mel_fb", torch.FloatTensor(mel_fb))
        self.register_buffer("window", torch.hann_window(N_FFT))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: [B, T] → mel: [B, N_MELS, frames]"""
        B = audio.shape[0]
        stft = torch.stft(
            audio, n_fft=N_FFT, hop_length=HOP_LENGTH,
            win_length=N_FFT, window=self.window,
            return_complex=True,
        )
        mag = stft.abs().clamp(min=1e-8)                    # [B, F, T]
        mel = torch.matmul(self.mel_fb, mag)                # [B, M, T]
        return torch.log(mel + 1e-8)


# ---------------------------------------------------------------------------
# Encoder — extracts pitch + timbre from mel spectrogram
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Convolutional encoder that extracts:
    - f0 (fundamental frequency) per frame
    - loudness per frame
    - z (timbre latent) per frame
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, (3, 3), stride=(2, 1), padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, (3, 3), stride=(2, 1), padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, (3, 3), stride=(2, 1), padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, (N_MELS // 8, 1), padding=0), nn.LeakyReLU(0.2),
        )
        self.gru = nn.GRU(512, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.f0_head      = nn.Linear(512, 1)
        self.ld_head      = nn.Linear(512, 1)
        self.z_mu_head    = nn.Linear(512, LATENT_DIM)
        self.z_logvar_head= nn.Linear(512, LATENT_DIM)

    def forward(self, mel: torch.Tensor):
        """mel: [B, M, T] → f0, ld, z_mu, z_logvar: [B, T, 1/1/D/D]"""
        x = mel.unsqueeze(1)                                 # [B, 1, M, T]
        x = self.conv(x)                                     # [B, 512, 1, T]
        x = x.squeeze(2).permute(0, 2, 1)                   # [B, T, 512]
        x, _ = self.gru(x)                                   # [B, T, 512]
        f0      = torch.sigmoid(self.f0_head(x))             # [B, T, 1] → 0-1
        ld      = self.ld_head(x)                            # [B, T, 1]
        z_mu    = self.z_mu_head(x)                          # [B, T, D]
        z_logvar= self.z_logvar_head(x)                      # [B, T, D]
        return f0, ld, z_mu, z_logvar


# ---------------------------------------------------------------------------
# Decoder — harmonic + noise synthesizer
# ---------------------------------------------------------------------------

class HarmonicOscillator(nn.Module):
    """
    Generates audio from fundamental frequency + harmonic amplitudes.
    Models a plucked string / blown reed as a sum of sinusoids.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        f0:   torch.Tensor,   # [B, T, 1] normalized 0-1
        amps: torch.Tensor,   # [B, T, N_HARMONICS]
    ) -> torch.Tensor:
        """Returns audio [B, samples]"""
        B, T, _ = f0.shape
        samples  = T * HOP_LENGTH

        # Convert normalized f0 to Hz (50Hz - 2000Hz range)
        f0_hz = 50.0 * (2000.0 / 50.0) ** f0.squeeze(-1)   # [B, T]

        # Upsample to audio rate
        f0_audio = F.interpolate(
            f0_hz.unsqueeze(1), size=samples, mode="linear", align_corners=False,
        ).squeeze(1)                                          # [B, samples]

        amps_audio = F.interpolate(
            amps.permute(0, 2, 1), size=samples,
            mode="linear", align_corners=False,
        ).permute(0, 2, 1)                                   # [B, samples, N_H]

        # Generate harmonics
        t = torch.linspace(0, samples / SR, samples, device=f0.device)
        audio = torch.zeros(B, samples, device=f0.device)

        for h in range(1, N_HARMONICS + 1):
            # Phase accumulation for each harmonic
            phase = 2 * math.pi * h * torch.cumsum(f0_audio / SR, dim=-1)
            harmonic = torch.sin(phase) * amps_audio[..., h - 1]
            audio += harmonic

        return audio


class NoiseFilter(nn.Module):
    """
    Filtered noise component — models breath, bow noise, string buzz.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        noise_mags: torch.Tensor,   # [B, T, N_NOISE]
    ) -> torch.Tensor:
        B, T, _ = noise_mags.shape
        samples  = T * HOP_LENGTH

        # White noise
        noise = torch.randn(B, samples, device=noise_mags.device)

        # Upsample filter magnitudes
        mags_audio = F.interpolate(
            noise_mags.permute(0, 2, 1), size=samples,
            mode="linear", align_corners=False,
        ).squeeze(1) if N_NOISE == 1 else F.interpolate(
            noise_mags.permute(0, 2, 1), size=samples,
            mode="linear", align_corners=False,
        ).permute(0, 2, 1)

        # Apply frequency-domain filter via STFT
        noise_stft = torch.stft(
            noise, n_fft=N_FFT, hop_length=HOP_LENGTH,
            return_complex=True,
        )                                                    # [B, F, T]

        # Interpolate mags to FFT bins
        filter_mags = F.interpolate(
            noise_mags.permute(0, 2, 1),
            size=noise_stft.shape[-1],
            mode="linear", align_corners=False,
        )                                                    # [B, N_NOISE, T]

        filter_mags = F.interpolate(
            filter_mags,
            size=noise_stft.shape[1],
            mode="linear", align_corners=False,
        ).permute(0, 2, 1).unsqueeze(0)

        filtered = noise_stft * torch.sigmoid(
            F.interpolate(
                noise_mags.permute(0, 2, 1),
                size=noise_stft.shape[1],
                mode="linear", align_corners=False,
            ).permute(0, 2, 1)
        ).unsqueeze(-1).expand_as(noise_stft)

        # ISTFT back to audio
        filtered_audio = torch.istft(
            filtered, n_fft=N_FFT, hop_length=HOP_LENGTH,
            length=samples,
        )
        return filtered_audio


class Decoder(nn.Module):
    """
    Generates audio from (f0, loudness, z_latent).
    Uses a GRU to model temporal dynamics, then harmonic + noise synthesis.
    """

    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(1 + 1 + LATENT_DIM, 512)
        self.gru = nn.GRU(512, 512, num_layers=3, batch_first=True)
        self.norm = nn.LayerNorm(512)

        # Output heads
        self.amp_head       = nn.Linear(512, N_HARMONICS)
        self.harm_dist_head = nn.Linear(512, N_HARMONICS)
        self.noise_head     = nn.Linear(512, N_NOISE)
        self.gain_head      = nn.Linear(512, 1)

        self.harmonic = HarmonicOscillator()

    def forward(
        self,
        f0: torch.Tensor,    # [B, T, 1]
        ld: torch.Tensor,    # [B, T, 1]
        z:  torch.Tensor,    # [B, T, D]
    ) -> torch.Tensor:

        x = torch.cat([f0, ld, z], dim=-1)                  # [B, T, 2+D]
        x = F.leaky_relu(self.input_proj(x), 0.2)
        x, _ = self.gru(x)
        x = self.norm(x)

        # Harmonic amplitudes — softmax so they sum to 1, scaled by overall amp
        overall_amp   = torch.sigmoid(self.amp_head(x))     # [B, T, N_H]
        harm_dist     = torch.softmax(self.harm_dist_head(x), dim=-1)
        amps          = overall_amp * harm_dist              # [B, T, N_H]

        # Loudness envelope
        gain = torch.sigmoid(self.gain_head(x))             # [B, T, 1]
        amps = amps * gain * torch.exp(ld / 20.0)

        # Harmonic audio
        audio = self.harmonic(f0, amps)                      # [B, samples]

        return audio


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class NeuralInstrument(nn.Module):
    """
    Full VAE-based neural instrument model.
    Encoder → (f0, ld, z) → Decoder → audio
    """

    def __init__(self):
        super().__init__()
        self.mel_spec = MelSpec()
        self.encoder  = Encoder()
        self.decoder  = Decoder()

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, audio: torch.Tensor):
        mel                   = self.mel_spec(audio)
        f0, ld, z_mu, z_logvar = self.encoder(mel)
        z                     = self.reparameterize(z_mu, z_logvar)
        recon                 = self.decoder(f0, ld, z)
        return recon, f0, ld, z_mu, z_logvar

    def synthesize(
        self,
        midi_note:    int,
        duration_sec: float = 2.0,
        dynamic:      float = 0.7,   # 0=soft, 1=loud
        vibrato:      float = 0.3,   # 0=none, 1=strong
    ) -> np.ndarray:
        """
        Synthesize a single note without needing a reference audio.
        Uses the decoder directly with crafted f0/ld/z inputs.
        """
        self.eval()
        with torch.no_grad():
            T = int(duration_sec * SR / HOP_LENGTH)

            # F0 trajectory
            hz = librosa.midi_to_hz(midi_note)
            f0_norm = math.log(hz / 50.0) / math.log(2000.0 / 50.0)
            f0_norm = max(0.01, min(0.99, f0_norm))

            f0 = torch.full((1, T, 1), f0_norm)
            # Add vibrato
            if vibrato > 0:
                vib_rate = 5.5   # Hz
                vib_depth = vibrato * 0.008
                t = torch.linspace(0, duration_sec, T)
                vib = vib_depth * torch.sin(2 * math.pi * vib_rate * t)
                f0 = f0 + vib.unsqueeze(0).unsqueeze(-1)
                f0 = f0.clamp(0.01, 0.99)

            # Loudness with natural envelope
            ld = torch.zeros(1, T, 1)
            attack  = max(1, int(T * 0.03))   # 3% attack
            release = max(1, int(T * 0.25))   # 25% release
            ld[:, :attack, 0]  = torch.linspace(-40, 0, attack)
            ld[:, attack:T - release, 0] = dynamic * 6 - 3
            ld[:, T - release:, 0] = torch.linspace(dynamic * 6 - 3, -40, release)

            # Zero latent (mean timbre)
            z = torch.zeros(1, T, LATENT_DIM)

            audio = self.decoder(f0, ld, z).squeeze(0).numpy()
            # Normalize
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.85
            return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class MultiScaleSpectralLoss(nn.Module):
    """
    Multi-scale spectral loss — compares spectrograms at multiple resolutions.
    More perceptually meaningful than raw waveform MSE.
    """

    def __init__(self):
        super().__init__()
        self.fft_sizes = [2048, 1024, 512, 256, 128]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for n_fft in self.fft_sizes:
            hop = n_fft // 4
            win = torch.hann_window(n_fft, device=pred.device)

            pred_stft = torch.stft(pred, n_fft=n_fft, hop_length=hop,
                                   window=win, return_complex=True).abs()
            tgt_stft  = torch.stft(target, n_fft=n_fft, hop_length=hop,
                                   window=win, return_complex=True).abs()

            # Log magnitude loss
            log_loss = F.l1_loss(
                torch.log(pred_stft.clamp(min=1e-8)),
                torch.log(tgt_stft.clamp(min=1e-8)),
            )
            # Linear magnitude loss
            lin_loss = F.l1_loss(pred_stft, tgt_stft) / (tgt_stft.mean() + 1e-8)
            loss += log_loss + lin_loss

        return loss / len(self.fft_sizes)


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(instrument: str, resume: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"  Training Neural Instrument: {instrument}")
    print(f"{'='*60}\n")

    model_dir = MODELS / instrument
    model_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset = InstrumentDataset(instrument, augment=True)
    if len(dataset) == 0:
        print(f"  No audio found for {instrument}.")
        print(f"  Add files to samples/{instrument}/ or samples/raw/{instrument}/")
        return

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True,
        num_workers=0, pin_memory=False,
    )

    # Model
    model     = NeuralInstrument()
    spec_loss = MultiScaleSpectralLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-5,
    )

    start_epoch = 0
    best_loss   = float("inf")

    # Resume from checkpoint
    ckpt_path = model_dir / "checkpoint.pt"
    if resume and ckpt_path.exists():
        print(f"  Resuming from checkpoint...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss   = ckpt.get("best_loss", float("inf"))
        print(f"  Resuming from epoch {start_epoch}, best loss {best_loss:.4f}")

    n_epochs = 100
    kl_weight = 0.0   # warm up KL over first 50 epochs

    print(f"  Training for {n_epochs} epochs on {len(dataset)} chunks")
    print(f"  Checkpoints saved to {model_dir}")
    print(f"  Press Ctrl+C to stop — training resumes with --resume\n")

    t0 = time.time()

    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss   = 0.0
        epoch_spec   = 0.0
        epoch_kl     = 0.0
        n_batches    = 0

        # KL warmup: ramp from 0 to 0.01 over first 50 epochs
        kl_weight = min(0.01, epoch / 50 * 0.01)

        for batch in loader:
            audio = batch                                     # [B, T]

            optimizer.zero_grad()
            recon, f0, ld, z_mu, z_logvar = model(audio)

            # Trim recon to match target length
            min_len = min(recon.shape[-1], audio.shape[-1])
            s_loss  = spec_loss(recon[..., :min_len], audio[..., :min_len])
            kl      = kl_loss(z_mu, z_logvar)
            loss    = s_loss + kl_weight * kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_spec += s_loss.item()
            epoch_kl   += kl.item()
            n_batches  += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_spec = epoch_spec / max(n_batches, 1)
        avg_kl   = epoch_kl   / max(n_batches, 1)
        elapsed  = (time.time() - t0) / 60

        # Progress
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            remaining = (elapsed / max(epoch - start_epoch + 1, 1)) * (n_epochs - epoch - 1)
            print(f"  Epoch {epoch+1:4d}/{n_epochs}  "
                  f"loss={avg_loss:.4f}  spec={avg_spec:.4f}  kl={avg_kl:.4f}  "
                  f"elapsed={elapsed:.1f}m  remaining≈{remaining:.1f}m")

        # Save checkpoint every 50 epochs
        if epoch % 50 == 0 or epoch == n_epochs - 1:
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "best_loss":  best_loss,
                "config": {
                    "sr": SR, "hop": HOP_LENGTH, "n_fft": N_FFT,
                    "n_mels": N_MELS, "latent_dim": LATENT_DIM,
                    "n_harmonics": N_HARMONICS,
                },
            }, ckpt_path)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_dir / "model_best.pt")

        # Generate a test note every 100 epochs to hear progress
        if epoch % 100 == 99:
            _save_test_note(model, instrument, model_dir, epoch)

    # Final save
    torch.save(model.state_dict(), model_dir / "model_final.pt")
    print(f"\n  Training complete.")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Model saved to {model_dir}/model_best.pt")
    print(f"  The audio engine will use this model automatically.")


def _save_test_note(model, instrument, model_dir, epoch):
    """Generate a test note and save it so you can hear training progress."""
    model.eval()
    try:
        # Synthesize middle C (MIDI 60)
        audio = model.synthesize(midi_note=60, duration_sec=2.0)
        out_path = model_dir / f"test_epoch_{epoch+1:04d}_C4.wav"
        sf.write(str(out_path), audio, SR)
        print(f"  Test note saved: {out_path.name}")
    except Exception as e:
        print(f"  Test note failed: {e}")
    model.train()


# ---------------------------------------------------------------------------
# Synthesis — called by indian_audio_engine
# ---------------------------------------------------------------------------

def load_model(instrument: str) -> NeuralInstrument | None:
    """Load trained model for an instrument. Returns None if not available."""
    model_dir = MODELS / instrument
    best      = model_dir / "model_best.pt"
    final     = model_dir / "model_final.pt"
    ckpt      = model_dir / "checkpoint.pt"

    path = best if best.exists() else final if final.exists() else None
    if path is None:
        return None

    try:
        model = NeuralInstrument()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        print(f"[neural] Loaded {instrument} model from {path.name}")
        return model
    except Exception as e:
        print(f"[neural] Failed to load {instrument} model: {e}")
        return None


def model_available(instrument: str) -> bool:
    model_dir = MODELS / instrument
    return (model_dir / "model_best.pt").exists() or \
           (model_dir / "model_final.pt").exists()


def synthesize_note(
    instrument:   str,
    midi_note:    int,
    duration_sec: float = 2.0,
    dynamic:      float = 0.7,
) -> np.ndarray | None:
    """
    Synthesize one note using the trained model.
    Called by indian_audio_engine._render_melody().
    """
    model = load_model(instrument)
    if model is None:
        return None
    return model.synthesize(midi_note, duration_sec, dynamic=dynamic)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural Indian instrument model")
    parser.add_argument("--instrument", type=str, required=True,
                        choices=["veena", "bansuri", "sitar", "carnatic_violin"],
                        help="Which instrument to train")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--test", action="store_true",
                        help="Generate test notes from existing model (no training)")
    args = parser.parse_args()

    if args.test:
        model = load_model(args.instrument)
        if model is None:
            print(f"No trained model found for {args.instrument}")
            print(f"Run: py -3.11 train_instrument.py --instrument {args.instrument}")
        else:
            out_dir = MODELS / args.instrument
            for midi in [48, 52, 55, 60, 64, 67, 72]:   # C3 E3 G3 C4 E4 G4 C5
                note = librosa.midi_to_note(midi, unicode=False)
                audio = model.synthesize(midi, duration_sec=2.0)
                path  = out_dir / f"test_{note}.wav"
                sf.write(str(path), audio, SR)
                print(f"  Generated: {path.name}")
            print(f"\nTest notes saved to {out_dir}/")
            print("Listen to them to evaluate model quality.")
    else:
        train(args.instrument, resume=args.resume)
