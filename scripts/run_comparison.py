import os
import glob
import cv2
import matplotlib.pyplot as plt
from src.pipeline import FoodQualityPipeline

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

pipeline = FoodQualityPipeline()
image_paths = glob.glob(os.path.join(RAW_DIR, "*.*"))

if not image_paths:
    print(f"Uyarı: {RAW_DIR} klasöründe görsel bulunamadı!")
else:
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue

        results = pipeline.run(img)
        filename = os.path.basename(path)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(results["original_rgb"])
        axes[0].set_title("Orijinal (RGB)")
        axes[0].axis("off")

        axes[1].imshow(results["enhanced"], cmap="gray")
        axes[1].set_title("İyileştirilmiş (HSI-I + HistEq)")
        axes[1].axis("off")

        axes[2].imshow(results["final_mask"], cmap="gray")
        axes[2].set_title("İkili Maske (Otsu + Morfoloji)")
        axes[2].axis("off")

        out_path = os.path.join(OUT_DIR, f"compare_{filename}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Kaydedildi: {out_path}")