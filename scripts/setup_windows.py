"""Install the Windows runtime from pinned upstream packages."""
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "Runtime"
CACHE = ROOT / ".downloads"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, digest):
    CACHE.mkdir(exist_ok=True)
    target = CACHE / url.rsplit("/", 1)[-1]
    if target.exists() and sha256(target) == digest:
        return target
    print(f"Downloading {target.name} ...", flush=True)
    with tempfile.TemporaryDirectory(dir=CACHE) as temp:
        partial = Path(temp) / "download"
        subprocess.run(["curl.exe", "--fail", "--location", "--silent", "--show-error",
                        "--retry", "3", "--connect-timeout", "20", "--max-time", "600",
                        "--output", str(partial), url], check=True)
        if sha256(partial) != digest:
            raise RuntimeError(f"Checksum mismatch for {target.name}. Run setup again.")
        partial.replace(target)
    return target


def install():
    if os.name != "nt" or sys.version_info[:2] != (3, 11) or struct.calcsize("P") != 8:
        raise SystemExit("Setup requires 64-bit Python 3.11 on Windows.")
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(ROOT / ".venv")], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check",
                    "--only-binary=:all:", "-r", str(ROOT / "requirements.lock.txt")], check=True)
    RUNTIME.mkdir(exist_ok=True)
    install_audio_engine()
    print("Setup complete. Open Start.cmd to launch the app.", flush=True)


def install_audio_engine():
    archive = download(
        "https://breakfastquay.com/files/releases/rubberband-4.0.0-gpl-executable-windows.zip",
        "f2d47fc64dbb42f6cc62edf7933ac4fa89d8f0ef8b9cf97b6afc263a7fe05644")
    with zipfile.ZipFile(archive) as distribution:
        distribution.extractall(RUNTIME / "rubberband")


if __name__ == "__main__":
    install()
