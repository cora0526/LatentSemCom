"""預先建立 EuroSAT=0 / DeepGlobe=1 的原始 latent 檢索表。
執行：python build_latent_database.py
依賴：torch、torchvision、Pillow（沿用原本可跑比較程式的環境）。
只執行 Encoder，使用 posterior mean；不混合、不加雜訊、不載入 DDPM。
注意：會收錄指定資料夾的全部匹配圖片，不會自動判定訓練／測試集。
未來測試時請排除已收錄圖片；若要避免更嚴格的資料洩漏，還需確認
測試圖片沒有參與原模型訓練，且同源衛星大圖的裁切不跨資料分割。

輸出 latent_database.pt 可如此讀取：
    bank = torch.load('save_data/latent_database/latent_database.pt',
                      map_location='cpu', weights_only=True)
    z_e = bank['latents'][bank['labels'] == 0]
    z_d = bank['latents'][bank['labels'] == 1]
這裡的 labels 是資料集身分，不是 EuroSAT 的土地類別。
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# 可直接修改以下預設路徑，或透過命令列參數覆寫。
EUROSAT_ROOT = r'D:/Python/SemCom/saved_data/EuroSAT/EuroSAT'
DEEPGLOBE_ROOT = r'D:/Python/SemCom/saved_data/deepglobe_64/train'
CHECKPOINT = './save_model/vae_gan_from_scratch_best.pth'
SAVE_ROOT = './save_data/latent_database'

class SharedVAEEncoder(nn.Module):
    def __init__(self, z_dim: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        feature_dim = 512 * 4 * 4
        self.fc_mu = nn.Linear(feature_dim, z_dim)
        self.fc_logvar = nn.Linear(feature_dim, z_dim)

    def forward(self, x: torch.Tensor):
        h = self.features(x).flatten(start_dim=1)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -10.0, 10.0)
        return mu, logvar



class ImagePaths(Dataset):
    def __init__(self, paths):
        self.paths = paths
        # 與原比較程式完全相同的前處理。
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert('RGB'))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eurosat-root', default=EUROSAT_ROOT)
    parser.add_argument('--deepglobe-root', default=DEEPGLOBE_ROOT)
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    parser.add_argument('--save-root', default=SAVE_ROOT)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error('batch-size must be positive; num-workers must be nonnegative.')
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    out = Path(args.save_root).resolve()
    names = ['latent_database.pt', 'latent_table.csv', 'metadata.json']
    if not args.overwrite and any((out / name).exists() for name in names):
        raise FileExistsError(f'Output already exists in {out}. Use --overwrite to replace.')

    # 和原程式相同：EuroSAT 是 ImageFolder，DeepGlobe 匹配 *_sat_*.jpg。
    euro = datasets.ImageFolder(args.eurosat_root)
    paths_e = [str(Path(path).resolve()) for path, _ in euro.samples]
    paths_d = [str(path.resolve()) for path in
               sorted(Path(args.deepglobe_root).glob('*_sat_*.jpg')) if path.is_file()]
    if not paths_d:
        raise RuntimeError(f'No *_sat_*.jpg images in {args.deepglobe_root}')
    paths = paths_e + paths_d
    labels = torch.tensor([0] * len(paths_e) + [1] * len(paths_d), dtype=torch.long)

    # 只需要 checkpoint 中的 encoder，不需要 generator 或 DDPM。
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if 'encoder' not in checkpoint:
        raise KeyError("Checkpoint must contain the 'encoder' state dictionary.")
    state = checkpoint['encoder']
    z_dim = int(state['fc_mu.weight'].shape[0])
    encoder = SharedVAEEncoder(z_dim)
    encoder.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    encoder.to(device).eval()
    encoder.requires_grad_(False)
    loader = DataLoader(ImagePaths(paths), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    latents = torch.empty((len(paths), z_dim), dtype=torch.float32)
    offset = 0
    print(f'EuroSAT (0): {len(paths_e)}; DeepGlobe (1): {len(paths_d)}')
    print(f'Device: {device}; latent dimension: {z_dim}')
    with torch.inference_mode():
        for batch_index, images in enumerate(loader):
            mu, _ = encoder(images.to(device, non_blocking=True))
            if not torch.isfinite(mu).all():
                raise RuntimeError(f'Non-finite latent in batch {batch_index}')
            count = mu.shape[0]
            latents[offset:offset + count] = mu.cpu().float()
            offset += count
            if batch_index % 20 == 0 or offset == len(paths):
                print(f'Encoded {offset}/{len(paths)}')

    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint_path': str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256': sha256(args.checkpoint),
        'label_mapping': {'0': 'EuroSAT', '1': 'DeepGlobe'},
        'eurosat_count': len(paths_e), 'deepglobe_count': len(paths_d),
        'z_dim': z_dim, 'representation': 'posterior mean (mu), float32',
        'preprocessing': 'RGB; Resize(64,64); ToTensor; Normalize(mean=0.5,std=0.5)',
        'latent_normalization': 'none', 'channel': 'none',
        'selection': 'all matching images in supplied roots; no automatic train/test split',
        'eurosat_root': str(Path(args.eurosat_root).resolve()),
        'deepglobe_root': str(Path(args.deepglobe_root).resolve()),
        'torch_version': str(torch.__version__),
    }
    out.mkdir(parents=True, exist_ok=True)
    # .pt 保留完整 float32，後续最近鄰搜尋應使用此檔。
    torch.save({'latents': latents, 'labels': labels, 'image_paths': paths,
                'metadata': metadata}, out / names[0])
    # CSV 一列對應一張圖片，包含全部 latent 維度，方便查看。
    with (out / names[1]).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['row_id', 'dataset', 'label', 'image_path'] +
                        [f'z_{i:03d}' for i in range(z_dim)])
        for i, path in enumerate(paths):
            label = int(labels[i])
            writer.writerow([i, 'EuroSAT' if label == 0 else 'DeepGlobe', label, path]
                            + latents[i].tolist())
    (out / names[2]).write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                               encoding='utf-8')
    for name in names:
        print('Saved:', out / name)
    print('Done. Use the same encoder/preprocessing and unnormalized mu for future queries.')


if __name__ == '__main__':
    main()
