import os
import requests
import sys

def download_weights():
    
    url = "https://github.com/sayanc227/RF-DETR-Infrastructure-Detection/releases/download/v1.0/checkpoint_best_ema.pth"
    
    output_dir = "weights"
    filename = "checkpoint_best_ema.pth"
    destination = os.path.join(output_dir, filename)

    if os.path.exists(destination):
        print(f"✅ Model weights already exist at {destination}")
        return

    print(f"⬇️ Downloading pre-trained RF-DETR model to {destination}...")
    print("   (This may take a few minutes depending on your internet speed)")
    
    os.makedirs(output_dir, exist_ok=True)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("✅ Download complete!")
        print(f"   Usage: python src/predict.py --checkpoint {destination} ...")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("   Please download the weights manually from the 'Releases' tab on GitHub.")

if __name__ == "__main__":
    download_weights()
