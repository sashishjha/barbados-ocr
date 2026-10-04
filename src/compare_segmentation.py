import os
import cv2
import pandas as pd
import numpy as np

def analyze_crops(csv_file, img_dir, sample_size=1000):
    df = pd.read_csv(csv_file)
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    heights, widths, aspect_ratios = [], [], []
    
    for _, row in sample.iterrows():
        img_id = str(row["ID"])
        for ext in [".jpg", ".jpeg", ".png"]:
            img_path = os.path.join(img_dir, f"{img_id}{ext}")
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    h, w = img.shape
                    heights.append(h)
                    widths.append(w)
                    aspect_ratios.append(w/h if h > 0 else 0)
                break
                
    return {
        "mean_h": np.mean(heights),
        "std_h": np.std(heights),
        "mean_w": np.mean(widths),
        "mean_ar": np.mean(aspect_ratios)
    }

print("=== Analyzing Train Images ===")
train_stats = analyze_crops("data/Train.csv", "data/images/")
print(train_stats)

print("\n=== Analyzing Test Images ===")
test_stats = analyze_crops("data/Test.csv", "data/images/")
print(test_stats)
