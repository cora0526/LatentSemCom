import rasterio
import numpy as np
from PIL import Image

path = r"D:/user/Downloads/archive_1/EuroSATallBands/SeaLake/SeaLake_993.tif"
with rasterio.open(path) as src:
    img = src.read([4, 3, 2])  # 常見 Sentinel-2 RGB: B4, B3, B2

img = np.transpose(img, (1, 2, 0))  # C,H,W -> H,W,C

img = img.astype(np.float32)
img = img / img.max() * 255
img = img.astype(np.uint8)

Image.fromarray(img).show()