from PIL import Image
import os
import glob

root = r"D:/Python/SemCom/saved_data/deepglobe"
splits = ["train", "valid", "test"]
save_path = r"D:/Python/SemCom/saved_data/deepglobe_64"

patch_size = 64
stride = 64

for split in splits:
    src_dir = os.path.join(root, split)
    save_dir = os.path.join(save_path, split)
    os.makedirs(save_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(src_dir, "*_sat.jpg")))

    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        base = os.path.splitext(os.path.basename(img_path))[0]

        idx = 0
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = img.crop((x, y, x + patch_size, y + patch_size))
                patch.save(os.path.join(save_dir, f"{base}_x{x:04d}_y{y:04d}.jpg"))
                idx += 1

        print(f"{split} | {base}: {idx} patches")

print("Done.")