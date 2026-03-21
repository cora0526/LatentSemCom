from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

save_dir = "mnist_jpg"
os.makedirs(save_dir, exist_ok=True)

dataset = datasets.MNIST(
    root="D:/Python/SemCom/saved_data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

for i in range(len(dataset)):
    img, label = dataset[i]
    save_image(img, f"{save_dir}/{i}_{label}.jpg")