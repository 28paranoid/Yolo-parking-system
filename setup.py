# setup.py (Cross-Platform)
# NOTE: OCR / license-plate components (easyocr, license_plate_detector.pt) removed.
# They were never used by the detection/occupancy pipeline.
import os
import subprocess


def run_pip(package):
    print(f"📦 Installing {package}...")
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", package])


def download_file(url, filename):
    if os.path.exists(filename):
        print(f"✅ {filename} already exists.")
        return
    print(f"⬇️ Downloading {filename}...")
    import requests
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"✅ Downloaded {filename}")
    else:
        print(f"❌ Failed to download {filename}")


if __name__ == "__main__":
    print("--- 1. Installing Dependencies ---")
    # Core libraries only (OCR removed)
    required_libs = [
        "ultralytics", "shapely", "boxmot==10.0.21",
        "requests", "pandas", "matplotlib"
    ]

    for lib in required_libs:
        try:
            run_pip(lib)
        except Exception as e:
            print(f"⚠️ Warning: Could not install {lib}. You may need C++ Build Tools. ({e})")

    print("\n--- 2. Downloading Pretrained Vehicle Model ---")
    # Pretrained COCO model -> used for robust VEHICLE detection in the hybrid pipeline.
    download_file(
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8n.pt"
    )

    print("\nℹ️  Your CUSTOM parking model ('best.pt') is produced by training on your")
    print("   Label-Studio-annotated Phuentsholing dataset. Place it next to main.py.")
    print("\n✅ Setup complete. Run 'main.py' next.")
