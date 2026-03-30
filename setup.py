"""
DriveSense Setup — downloads the dlib 68-point facial landmark predictor
and verifies all dependencies are installed.
"""

import sys
import os
import subprocess
import urllib.request

MODEL_URL  = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
MODEL_BZ2  = "shape_predictor_68_face_landmarks.dat.bz2"
MODEL_DAT  = "shape_predictor_68_face_landmarks.dat"

REQUIRED = ["opencv-python", "dlib", "numpy", "scipy", "imutils"]


def run(cmd):
    print(f"  $ {cmd}")
    subprocess.check_call(cmd, shell=True)


def check_install():
    print("\n[1/3] Checking / installing Python dependencies…")
    for pkg in REQUIRED:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"  ✓  {pkg}")
        except ImportError:
            print(f"  ↳  Installing {pkg}…")
            run(f"{sys.executable} -m pip install {pkg}")


def download_model():
    print("\n[2/3] Downloading dlib facial landmark predictor…")
    if os.path.exists(MODEL_DAT):
        print(f"  ✓  {MODEL_DAT} already present – skipping download.")
        return

    if not os.path.exists(MODEL_BZ2):
        print(f"  Downloading {MODEL_URL}")
        def progress(count, block, total):
            pct = count * block / total * 100
            bar = "#" * int(pct / 2)
            print(f"\r  [{bar:<50}] {pct:.1f}%", end="", flush=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_BZ2, reporthook=progress)
        print()

    print(f"  Decompressing {MODEL_BZ2}…")
    import bz2, shutil
    with bz2.open(MODEL_BZ2, "rb") as f_in:
        with open(MODEL_DAT, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(MODEL_BZ2)
    print(f"  ✓  {MODEL_DAT} ready.")


def verify():
    print("\n[3/3] Verifying setup…")
    import cv2, dlib, numpy, scipy, imutils
    print(f"  OpenCV  : {cv2.__version__}")
    print(f"  dlib    : {dlib.__version__}")
    print(f"  NumPy   : {numpy.__version__}")
    print(f"  SciPy   : {scipy.__version__}")
    assert os.path.exists(MODEL_DAT), "Landmark predictor file missing!"
    print(f"  Predictor file : OK")
    print("\n✅  All good! Run DriveSense with:\n")
    print("     python drivesense.py\n")


if __name__ == "__main__":
    print("=" * 52)
    print("  DriveSense Setup")
    print("=" * 52)
    check_install()
    download_model()
    verify()
