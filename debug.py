import sys, os, traceback
sys.path.insert(0, os.getcwd())

print("python:", sys.version)

try:
    print("running app.py...")
    exec(open('app.py').read())
except SystemExit as e:
    print("SystemExit:", e)
    traceback.print_exc()
except Exception as e:
    print("Exception:", e)
    traceback.print_exc()

input("Press Enter...")