#!/usr/bin/env python
# launch.py

import os
import sys
import subprocess
import platform
from pathlib import Path

# -------------------------------------------------- config
REPO_URLS = [
    "https://github.com/ultralytics/ultralytics.git",
    "https://github.com/Trusted-AI/adversarial-robustness-toolbox.git",
]


TAG_OVERRIDES = {
    "https://github.com/Trusted-AI/adversarial-robustness-toolbox.git": "1.19.1",
}




PARENT_DIR       = Path("repository")
SUPPORTED_MINORS = range(8, 13)               
GIT              = os.environ.get("GIT", "git")


# -------------------------------------------------- utils
def check_python_version():
    major, minor, micro = sys.version_info[:3]
    if not (major == 3 and minor in SUPPORTED_MINORS):
        sys.exit(
            f"\nINCOMPATIBLE PYTHON VERSION\n"
            f"Script expects Python 3.{SUPPORTED_MINORS.start}–{SUPPORTED_MINORS.stop-1}, "
            f"but you have {major}.{minor}.{micro}.\n"
        )

def run(cmd: str):
    """Run a shell command; raise if it fails."""
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {cmd}")

# -------------------------------------------------- main logic
def clone_if_missing(url: str):
    repo_name = Path(url).stem            # "ultralytics"
    dest = PARENT_DIR / repo_name

    if dest.exists():
        print(f"[✓] Repo '{repo_name}' already exists – skip.")
        return dest

    print(f"[→] Cloning '{repo_name}' …")
    PARENT_DIR.mkdir(parents=True, exist_ok=True)
    run(f'"{GIT}" clone "{url}" "{dest}"')
    print(f"[✓] Done cloning '{repo_name}'.\n")
    return dest

def checkout_tag(repo_dir: Path, tag: str):
    print(f"[→] Checking out tag '{tag}' in {repo_dir.name} …")
    run(f'"{GIT}" -C "{repo_dir}" fetch --tags')
    run(f'"{GIT}" -C "{repo_dir}" checkout {tag}')
    print(f"[✓] Now at tag '{tag}' for {repo_dir.name}.\n")

def install_requirements(folder: Path | str = "."):
    folder = Path(folder)
    req_file = folder / "requirements.txt"
    if not req_file.is_file():
        print(f"[!] No requirements.txt found in {folder}.")
        return
    print(f"[→] Installing requirements from '{req_file}' …")
    run(f'"{sys.executable}" -m pip install -r "{req_file}"')
    print(f"[✓] Requirements installed for '{folder.name}'.\n")

def main():
    check_python_version()
    print(f"Python {platform.python_version()}\n")

    for url in REPO_URLS:
        try:
            repo_path = clone_if_missing(url)

            if url in TAG_OVERRIDES:
                checkout_tag(repo_path, TAG_OVERRIDES[url])

            install_requirements(repo_path)
        except Exception as e:
            print(f"[!] Failed to process {url}: {e}")

    install_requirements(".")
    run(f'"{sys.executable}" -m pip install --no-deps -r requirements_local.txt')
    print("All done.")

if __name__ == "__main__":
    main()
