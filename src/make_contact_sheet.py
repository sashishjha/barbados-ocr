import pandas as pd
from PIL import Image
import os

test_csv = "data/Test.csv"
img_dir = "data/images/"
out_path = "report/diagnostics/contact_sheet.png"
os.makedirs("report/diagnostics", exist_ok=True)

df = pd.read_csv(test_csv)
sample = df.sample(min(100, len(df)), random_state=42)

cols, rows_grid = 5, 20
thumb_w, thumb_h = 300, 80
canvas = Image.new("RGB", (cols * thumb_w, rows_grid * thumb_h), (200, 200, 200))

found = 0
for i, (_, row) in enumerate(sample.iterrows()):
    img_id = str(row["ID"])
    for ext in [".jpg", ".jpeg", ".png"]:
        img_path = os.path.join(img_dir, f"{img_id}{ext}")
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((thumb_w, thumb_h))
                x = (found % cols) * thumb_w
                y = (found // cols) * thumb_h
                canvas.paste(img, (x, y))
                found += 1
            except Exception as e:
                pass
            break
canvas.save(out_path)
print(f"Saved contact sheet with {found} images to {out_path}")
