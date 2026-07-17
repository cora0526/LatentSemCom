import os
import warnings
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, features
from pyldpc import make_ldpc, decode, get_message
from pyldpc.utils import binaryproduct
from torchvision import datasets, transforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import pandas as pd

from skimage.metrics import structural_similarity as structural_ssim


# ============================================================
# Configuration
# ============================================================
DATA_ROOT = r"D:/Python/SemCom/saved_data"
SAVE_ROOT = r"D:/Python/SysTest/baseline_JPEG2000_MNIST"

NUM_IMAGES = 25
SNR_LIST = [30, 28, 26, 24, 22, 
            20, 18, 16, 14, 12, 
            10, 8,  6,  4,  2, 0]                 # Example: [20, 15, 10, 5, 0]
JPEG2000_COMPRESSION_RATIO = 2  # Approximate 8:1 compression
JPEG2000_IRREVERSIBLE = True    # True = lossy 9/7 wavelet
JPEG2000_NUM_RESOLUTIONS = 4    # Suitable for 28x28 images

LDPC_N = 2048
LDPC_DV = 2
LDPC_DC = 4
LDPC_MAX_ITER = 100
RANDOM_SEED = 42

# Fixed 2x2 MIMO channel
MIMO_H = np.array(
    [
        [1.00, 0.15],
        [0.10, 0.90],
    ],
    dtype=np.complex128,
)


# ============================================================
# Reproducibility
# ============================================================
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def calculate_ssim(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
):
    """
    original, reconstructed:
        Tensor [C,H,W], range [0,1]
    """

    original = original.detach().cpu().float().clamp(0, 1)
    reconstructed = reconstructed.detach().cpu().float().clamp(0, 1)

    if original.shape != reconstructed.shape:
        reconstructed = F.interpolate(
            reconstructed.unsqueeze(0),
            size=original.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    # 灰階影像
    if original.shape[0] == 1:
        original_np = original.squeeze(0).numpy()
        reconstructed_np = reconstructed.squeeze(0).numpy()

        return float(
            structural_ssim(
                original_np,
                reconstructed_np,
                data_range=1.0,
            )
        )

    # RGB影像
    original_np = original.permute(1, 2, 0).numpy()
    reconstructed_np = reconstructed.permute(1, 2, 0).numpy()

    return float(
        structural_ssim(
            original_np,
            reconstructed_np,
            channel_axis=2,
            data_range=1.0,
        )
    )

# ============================================================
# JPEG2000 encode / decode
# ============================================================
def tensor_to_jpeg2000_bits(
    x: torch.Tensor,
    compression_ratio: float = 8,
    irreversible: bool = True,
    num_resolutions: int = 4,
):
    """
    Convert an image tensor to a JPEG2000 bitstream.

    Args:
        x:
            Tensor [C, H, W], range [0, 1].
        compression_ratio:
            quality_mode="rates" parameter.
            Larger value means stronger compression.
        irreversible:
            True: lossy JPEG2000.
            False: reversible/lossless JPEG2000.
        num_resolutions:
            Number of wavelet resolutions. For 28x28 images, 4 is safer
            than the larger default value.

    Returns:
        bits: np.ndarray containing 0/1 bits.
        byte_len: original JPEG2000 byte length.
    """
    x = x.detach().cpu().clamp(0, 1)

    if x.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got shape {tuple(x.shape)}")

    channels = x.shape[0]

    if channels == 1:
        arr = (x.squeeze(0).numpy() * 255.0).round().astype(np.uint8)
        pil_img = Image.fromarray(arr, mode="L")
    elif channels == 3:
        arr = (
            x.permute(1, 2, 0).numpy() * 255.0
        ).round().astype(np.uint8)
        pil_img = Image.fromarray(arr, mode="RGB")
    else:
        raise ValueError(f"Unsupported channel count: {channels}")

    buffer = BytesIO()

    save_kwargs = {
        "format": "JPEG2000",
        "num_resolutions": int(num_resolutions),
        "irreversible": bool(irreversible),
    }

    if irreversible:
        save_kwargs.update(
            {
                "quality_mode": "rates",
                "quality_layers": [float(compression_ratio)],
            }
        )

    pil_img.save(buffer, **save_kwargs)

    byte_data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    bits = np.unpackbits(byte_data)

    return bits.astype(np.int64), len(byte_data)


def jpeg2000_bits_to_tensor(
    bits: np.ndarray,
    byte_len: int,
    image_size=(28, 28),
    channels: int = 1,
):
    """
    Convert received bits back into a JPEG2000 image tensor.

    Returns:
        tensor: [C,H,W], range [0,1].
        success: whether JPEG2000 decoding succeeded.
        error_message: empty string on success.
    """
    bits = np.asarray(bits[: byte_len * 8], dtype=np.uint8)

    try:
        byte_data = np.packbits(bits)
        buffer = BytesIO(byte_data.tobytes())

        img = Image.open(buffer)
        img.load()

        if channels == 1:
            img = img.convert("L")
        elif channels == 3:
            img = img.convert("RGB")
        else:
            raise ValueError(f"Unsupported channel count: {channels}")

        # PIL uses (width, height)
        target_size = (image_size[1], image_size[0])
        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.BILINEAR)

        arr = np.asarray(img, dtype=np.float32) / 255.0

        if channels == 1:
            tensor = torch.from_numpy(arr.copy()).unsqueeze(0)
        else:
            tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1)

        return tensor, True, ""

    except Exception as exc:
        tensor = torch.zeros(
            channels,
            image_size[0],
            image_size[1],
            dtype=torch.float32,
        )
        return tensor, False, str(exc)


# ============================================================
# QPSK modulation and soft demodulation
# ============================================================
def qpsk_mod(bits: np.ndarray):
    """
    Gray-independent QPSK mapping used by the original program:

        bit pair [b0,b1]
        symbol = ((1-2*b0) + j(1-2*b1)) / sqrt(2)

    Args:
        bits: 1-D array of 0/1 bits.

    Returns:
        complex QPSK symbols.
    """
    bits = np.asarray(bits, dtype=np.int64).reshape(-1)

    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)

    pairs = bits.reshape(-1, 2)

    symbols = (
        (1.0 - 2.0 * pairs[:, 0])
        + 1j * (1.0 - 2.0 * pairs[:, 1])
    ) / np.sqrt(2.0)

    return symbols.astype(np.complex128)


def qpsk_hard_demod(symbols: np.ndarray):
    """
    Hard decisions used only for pre-LDPC BER reporting.
    """
    bits_i = (np.real(symbols) < 0).astype(np.int64)
    bits_q = (np.imag(symbols) < 0).astype(np.int64)

    return np.stack([bits_i, bits_q], axis=1).reshape(-1)


def qpsk_to_bpsk_soft_observations(symbols: np.ndarray):
    """
    Convert equalized QPSK symbols into BPSK-like soft observations
    expected by pyldpc.decode().

    Since QPSK symbols have components +/-1/sqrt(2), multiplying each
    component by sqrt(2) restores BPSK amplitude +/-1.

    Output bit ordering matches qpsk_mod():
        Re(symbol0), Im(symbol0), Re(symbol1), Im(symbol1), ...
    """
    soft_i = np.real(symbols) * np.sqrt(2.0)
    soft_q = np.imag(symbols) * np.sqrt(2.0)

    return np.stack([soft_i, soft_q], axis=1).reshape(-1)


# ============================================================
# 2x2 MIMO + AWGN + ZF
# ============================================================
def mimo_2x2_zf_two_streams(
    symbols_m: np.ndarray,
    symbols_f: np.ndarray,
    snr_db: float,
    channel_matrix: np.ndarray = MIMO_H,
):
    """
    Transmit two independent QPSK streams through a fixed 2x2 MIMO
    channel, add AWGN once, and apply zero-forcing equalization.

    Returns:
        symbols_m_hat
        symbols_f_hat
        noise_var_m:
            Effective BPSK-observation noise variance for stream M.
        noise_var_f:
            Effective BPSK-observation noise variance for stream F.
    """
    original_len_m = len(symbols_m)
    original_len_f = len(symbols_f)
    max_len = max(original_len_m, original_len_f)

    symbols_m_pad = np.pad(
        symbols_m,
        (0, max_len - original_len_m),
        mode="constant",
    )
    symbols_f_pad = np.pad(
        symbols_f,
        (0, max_len - original_len_f),
        mode="constant",
    )

    x = np.stack([symbols_m_pad, symbols_f_pad], axis=0)

    h = np.asarray(channel_matrix, dtype=np.complex128)

    if h.shape != (2, 2):
        raise ValueError(f"MIMO channel must be 2x2, got {h.shape}")

    # Normalize Frobenius norm to sqrt(2).
    h = h / np.linalg.norm(h, "fro") * np.sqrt(2.0)

    y = h @ x

    signal_power = float(np.mean(np.abs(y) ** 2))
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    rx_complex_noise_var = signal_power / snr_linear

    noise = np.sqrt(rx_complex_noise_var / 2.0) * (
        np.random.randn(*y.shape)
        + 1j * np.random.randn(*y.shape)
    )

    y_noisy = y + noise

    w_zf = np.linalg.pinv(h)
    x_hat = w_zf @ y_noisy

    # Effective complex noise covariance after ZF.
    zf_noise_cov = (
        rx_complex_noise_var
        * (w_zf @ w_zf.conj().T)
    )

    complex_noise_var_m = max(
        float(np.real(zf_noise_cov[0, 0])),
        1e-12,
    )
    complex_noise_var_f = max(
        float(np.real(zf_noise_cov[1, 1])),
        1e-12,
    )

    # qpsk_to_bpsk_soft_observations multiplies each real component by
    # sqrt(2). Therefore its real-valued BPSK-like noise variance equals
    # the equalized complex noise variance.
    bpsk_noise_var_m = complex_noise_var_m
    bpsk_noise_var_f = complex_noise_var_f

    return (
        x_hat[0, :original_len_m],
        x_hat[1, :original_len_f],
        bpsk_noise_var_m,
        bpsk_noise_var_f,
    )


# ============================================================
# LDPC helpers
# ============================================================
def pad_bits(bits: np.ndarray, block_len: int):
    bits = np.asarray(bits, dtype=np.int64).reshape(-1)
    pad_len = (block_len - len(bits) % block_len) % block_len

    if pad_len == 0:
        return bits.copy(), 0

    return (
        np.concatenate(
            [bits, np.zeros(pad_len, dtype=np.int64)]
        ),
        pad_len,
    )


def ldpc_encode_bits(generator_matrix, message_bits: np.ndarray):
    """
    Pure LDPC encoding without adding channel noise.
    """
    coded_bits = binaryproduct(generator_matrix, message_bits)

    return np.asarray(
        coded_bits,
        dtype=np.int64,
    ).reshape(-1)


def safe_ldpc_decode(
    parity_check_matrix,
    generator_matrix,
    soft_observations: np.ndarray,
    effective_snr_db: float,
    max_iter: int,
):
    """
    Run pyldpc decoding and return message bits.

    If decoding raises an exception, return None and the error text.
    """
    try:
        decoded_codeword = decode(
            parity_check_matrix,
            soft_observations,
            effective_snr_db,
            maxiter=max_iter,
        )

        message = get_message(
            generator_matrix,
            decoded_codeword,
        )

        return (
            np.asarray(message, dtype=np.int64).reshape(-1),
            "",
        )

    except Exception as exc:
        return None, str(exc)


def ldpc_transmit_two_streams(
    bits_m: np.ndarray,
    bits_f: np.ndarray,
    h_ldpc,
    g_ldpc,
    snr_db: float,
    max_iter: int = 100,
):
    """
    JPEG2000 bits
        -> LDPC
        -> QPSK
        -> 2x2 MIMO
        -> AWGN
        -> ZF
        -> QPSK soft observations
        -> LDPC decoding

    Returns:
        recovered_m
        recovered_f
        statistics
    """
    bits_m = np.asarray(bits_m, dtype=np.int64).reshape(-1)
    bits_f = np.asarray(bits_f, dtype=np.int64).reshape(-1)

    k = int(g_ldpc.shape[1])

    bits_m_pad, _ = pad_bits(bits_m, k)
    bits_f_pad, _ = pad_bits(bits_f, k)

    num_blocks_m = len(bits_m_pad) // k
    num_blocks_f = len(bits_f_pad) // k
    num_blocks = max(num_blocks_m, num_blocks_f)

    bits_m_pad = np.pad(
        bits_m_pad,
        (0, num_blocks * k - len(bits_m_pad)),
        mode="constant",
    )
    bits_f_pad = np.pad(
        bits_f_pad,
        (0, num_blocks * k - len(bits_f_pad)),
        mode="constant",
    )

    blocks_m = bits_m_pad.reshape(num_blocks, k)
    blocks_f = bits_f_pad.reshape(num_blocks, k)

    recovered_m_blocks = []
    recovered_f_blocks = []

    total_coded_errors_m = 0
    total_coded_errors_f = 0
    total_coded_bits_m = 0
    total_coded_bits_f = 0

    total_message_errors_m = 0
    total_message_errors_f = 0
    total_message_bits_m = 0
    total_message_bits_f = 0

    failed_ldpc_blocks_m = 0
    failed_ldpc_blocks_f = 0

    for block_index, (message_m, message_f) in enumerate(
        zip(blocks_m, blocks_f)
    ):
        coded_bits_m = ldpc_encode_bits(g_ldpc, message_m)
        coded_bits_f = ldpc_encode_bits(g_ldpc, message_f)

        symbols_m = qpsk_mod(coded_bits_m)
        symbols_f = qpsk_mod(coded_bits_f)

        (
            symbols_m_hat,
            symbols_f_hat,
            noise_var_m,
            noise_var_f,
        ) = mimo_2x2_zf_two_streams(
            symbols_m,
            symbols_f,
            snr_db,
        )

        # Pre-LDPC BER.
        hard_bits_m = qpsk_hard_demod(symbols_m_hat)[
            : len(coded_bits_m)
        ]
        hard_bits_f = qpsk_hard_demod(symbols_f_hat)[
            : len(coded_bits_f)
        ]

        total_coded_errors_m += int(
            np.count_nonzero(hard_bits_m != coded_bits_m)
        )
        total_coded_errors_f += int(
            np.count_nonzero(hard_bits_f != coded_bits_f)
        )
        total_coded_bits_m += len(coded_bits_m)
        total_coded_bits_f += len(coded_bits_f)

        # Soft BPSK-like observations.
        soft_m = qpsk_to_bpsk_soft_observations(
            symbols_m_hat
        )[: len(coded_bits_m)]
        soft_f = qpsk_to_bpsk_soft_observations(
            symbols_f_hat
        )[: len(coded_bits_f)]

        # pyldpc assumes unit-amplitude BPSK and noise variance
        # 10^(-SNR/10).
        effective_snr_m_db = -10.0 * np.log10(noise_var_m)
        effective_snr_f_db = -10.0 * np.log10(noise_var_f)

        decoded_message_m, decode_error_m = safe_ldpc_decode(
            h_ldpc,
            g_ldpc,
            soft_m,
            effective_snr_m_db,
            max_iter,
        )
        decoded_message_f, decode_error_f = safe_ldpc_decode(
            h_ldpc,
            g_ldpc,
            soft_f,
            effective_snr_f_db,
            max_iter,
        )

        if decoded_message_m is None:
            failed_ldpc_blocks_m += 1
            decoded_message_m = np.zeros(k, dtype=np.int64)
            warnings.warn(
                f"Stream M block {block_index} decode failed: "
                f"{decode_error_m}"
            )

        if decoded_message_f is None:
            failed_ldpc_blocks_f += 1
            decoded_message_f = np.zeros(k, dtype=np.int64)
            warnings.warn(
                f"Stream F block {block_index} decode failed: "
                f"{decode_error_f}"
            )

        decoded_message_m = decoded_message_m[:k]
        decoded_message_f = decoded_message_f[:k]

        recovered_m_blocks.append(decoded_message_m)
        recovered_f_blocks.append(decoded_message_f)

        total_message_errors_m += int(
            np.count_nonzero(decoded_message_m != message_m)
        )
        total_message_errors_f += int(
            np.count_nonzero(decoded_message_f != message_f)
        )
        total_message_bits_m += k
        total_message_bits_f += k

    recovered_m = np.concatenate(
        recovered_m_blocks
    )[: len(bits_m)]
    recovered_f = np.concatenate(
        recovered_f_blocks
    )[: len(bits_f)]

    statistics = {
        "pre_ber_m": (
            total_coded_errors_m / total_coded_bits_m
            if total_coded_bits_m
            else 0.0
        ),
        "pre_ber_f": (
            total_coded_errors_f / total_coded_bits_f
            if total_coded_bits_f
            else 0.0
        ),
        "post_ber_padded_m": (
            total_message_errors_m / total_message_bits_m
            if total_message_bits_m
            else 0.0
        ),
        "post_ber_padded_f": (
            total_message_errors_f / total_message_bits_f
            if total_message_bits_f
            else 0.0
        ),
        "post_ber_m": float(np.mean(recovered_m != bits_m)),
        "post_ber_f": float(np.mean(recovered_f != bits_f)),
        "failed_ldpc_blocks_m": failed_ldpc_blocks_m,
        "failed_ldpc_blocks_f": failed_ldpc_blocks_f,
        "num_ldpc_blocks": num_blocks,
        "coded_bits_per_stream": num_blocks * h_ldpc.shape[1],
    }

    return (
        recovered_m.astype(np.int64),
        recovered_f.astype(np.int64),
        statistics,
    )


# ============================================================
# Metrics
# ============================================================
def calculate_psnr(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    max_value: float = 1.0,
):
    mse = F.mse_loss(
        reconstructed.float(),
        original.float(),
    ).item()

    if mse <= 0:
        return float("inf")

    return 10.0 * np.log10((max_value ** 2) / mse)


# ============================================================
# Full JPEG2000 + LDPC + MIMO baseline
# ============================================================
def jpeg2000_ldpc_mimo_baseline(
    x_m: torch.Tensor,
    x_f: torch.Tensor,
    h_ldpc,
    g_ldpc,
    snr_db: float = 10,
    compression_ratio: float = 2,
    irreversible: bool = True,
    num_resolutions: int = 4,
):
    channels_m, height_m, width_m = x_m.shape
    channels_f, height_f, width_f = x_f.shape

    bits_m, byte_len_m = tensor_to_jpeg2000_bits(
        x_m,
        compression_ratio=compression_ratio,
        irreversible=irreversible,
        num_resolutions=num_resolutions,
    )
    bits_f, byte_len_f = tensor_to_jpeg2000_bits(
        x_f,
        compression_ratio=compression_ratio,
        irreversible=irreversible,
        num_resolutions=num_resolutions,
    )

    (
        recovered_bits_m,
        recovered_bits_f,
        statistics,
    ) = ldpc_transmit_two_streams(
        bits_m,
        bits_f,
        h_ldpc,
        g_ldpc,
        snr_db,
        max_iter=LDPC_MAX_ITER,
    )

    reconstructed_m, success_m, error_m = (
        jpeg2000_bits_to_tensor(
            recovered_bits_m,
            byte_len_m,
            image_size=(height_m, width_m),
            channels=channels_m,
        )
    )
    reconstructed_f, success_f, error_f = (
        jpeg2000_bits_to_tensor(
            recovered_bits_f,
            byte_len_f,
            image_size=(height_f, width_f),
            channels=channels_f,
        )
    )

    statistics.update(
        {
            "source_bits_m": len(bits_m),
            "source_bits_f": len(bits_f),
            "source_bytes_m": byte_len_m,
            "source_bytes_f": byte_len_f,
            "jpeg2000_success_m": success_m,
            "jpeg2000_success_f": success_f,
            "jpeg2000_error_m": error_m,
            "jpeg2000_error_f": error_f,
            "psnr_m": calculate_psnr(x_m, reconstructed_m),
            "psnr_f": calculate_psnr(x_f, reconstructed_f),
        }
    )

    return reconstructed_m, reconstructed_f, statistics


# ============================================================
# Main experiment
# ============================================================
# ============================================================
# Main experiment
# ============================================================
def main():
    if not features.check("jpg_2000"):
        raise RuntimeError(
            "This Pillow installation does not support JPEG2000.\n"
            "Install Pillow/OpenJPEG, for example:\n"
            "  conda install -c conda-forge pillow openjpeg"
        )

    os.makedirs(SAVE_ROOT, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset_m = datasets.MNIST(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transform,
    )

    dataset_f = datasets.FashionMNIST(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transform,
    )

    # 建立一次 LDPC，所有圖片與 SNR 共用
    h_ldpc, g_ldpc = make_ldpc(
        LDPC_N,
        LDPC_DV,
        LDPC_DC,
        systematic=True,
        sparse=True,
        seed=RANDOM_SEED,
    )

    code_rate = g_ldpc.shape[1] / g_ldpc.shape[0]

    print("=" * 72)
    print("JPEG2000 + LDPC + QPSK + 2x2 MIMO + AWGN + ZF")
    print(f"JPEG2000 support : {features.check('jpg_2000')}")
    print(f"LDPC n, k        : {g_ldpc.shape[0]}, {g_ldpc.shape[1]}")
    print(f"LDPC code rate   : {code_rate:.4f}")
    print(f"MIMO H condition : {np.linalg.cond(MIMO_H):.4f}")
    print("=" * 72)

    # ========================================================
    # 儲存原圖
    # ========================================================
    originals_m = torch.stack([
        dataset_m[i][0]
        for i in range(NUM_IMAGES)
    ])

    originals_f = torch.stack([
        dataset_f[i][0]
        for i in range(NUM_IMAGES)
    ])

    save_image(
        originals_m,
        os.path.join(
            SAVE_ROOT,
            "original_MNIST_5x5.png",
        ),
        nrow=5,
        padding=2,
        normalize=False,
    )

    save_image(
        originals_f,
        os.path.join(
            SAVE_ROOT,
            "original_FashionMNIST_5x5.png",
        ),
        nrow=5,
        padding=2,
        normalize=False,
    )

    # ========================================================
    # 跨 SNR 統計
    # ========================================================
    snr_results = []

    mnist_psnr_mean_list = []
    mnist_ssim_mean_list = []

    fashion_psnr_mean_list = []
    fashion_ssim_mean_list = []

    per_image_results = []

    # ========================================================
    # 測試所有 SNR
    # ========================================================
    for snr_db in SNR_LIST:
        reconstructed_m_list = []
        reconstructed_f_list = []

        success_m_count = 0
        success_f_count = 0

        pre_ber_m_values = []
        pre_ber_f_values = []

        post_ber_m_values = []
        post_ber_f_values = []

        psnr_m_values = []
        psnr_f_values = []

        ssim_m_values = []
        ssim_f_values = []

        source_bytes_m_values = []
        source_bytes_f_values = []

        coded_bits_values = []

        print(f"\n{'=' * 72}")
        print(f"Testing SNR = {snr_db} dB")
        print(f"{'=' * 72}")

        for image_index in range(NUM_IMAGES):
            x_m, label_m = dataset_m[image_index]
            x_f, label_f = dataset_f[image_index]

            reconstructed_m, reconstructed_f, stats = (
                jpeg2000_ldpc_mimo_baseline(
                    x_m,
                    x_f,
                    h_ldpc,
                    g_ldpc,
                    snr_db=snr_db,
                    compression_ratio=(
                        JPEG2000_COMPRESSION_RATIO
                    ),
                    irreversible=JPEG2000_IRREVERSIBLE,
                    num_resolutions=(
                        JPEG2000_NUM_RESOLUTIONS
                    ),
                )
            )

            reconstructed_m_list.append(reconstructed_m)
            reconstructed_f_list.append(reconstructed_f)

            # JPEG2000 是否成功解碼
            success_m = bool(
                stats["jpeg2000_success_m"]
            )
            success_f = bool(
                stats["jpeg2000_success_f"]
            )

            success_m_count += int(success_m)
            success_f_count += int(success_f)

            # BER
            pre_ber_m = stats["pre_ber_m"]
            pre_ber_f = stats["pre_ber_f"]

            post_ber_m = stats["post_ber_m"]
            post_ber_f = stats["post_ber_f"]

            pre_ber_m_values.append(pre_ber_m)
            pre_ber_f_values.append(pre_ber_f)

            post_ber_m_values.append(post_ber_m)
            post_ber_f_values.append(post_ber_f)

            # PSNR
            psnr_m = calculate_psnr(
                x_m,
                reconstructed_m,
            )

            psnr_f = calculate_psnr(
                x_f,
                reconstructed_f,
            )

            # SSIM
            ssim_m = calculate_ssim(
                x_m,
                reconstructed_m,
            )

            ssim_f = calculate_ssim(
                x_f,
                reconstructed_f,
            )

            psnr_m_values.append(psnr_m)
            psnr_f_values.append(psnr_f)

            ssim_m_values.append(ssim_m)
            ssim_f_values.append(ssim_f)

            source_bytes_m_values.append(
                stats["source_bytes_m"]
            )

            source_bytes_f_values.append(
                stats["source_bytes_f"]
            )

            coded_bits_values.append(
                stats["coded_bits_per_stream"]
            )

            m_status = "OK" if success_m else "FAIL"
            f_status = "OK" if success_f else "FAIL"

            print(
                f"[{image_index + 1:02d}/{NUM_IMAGES}] "
                f"label M={label_m}, F={label_f} | "
                f"bytes M={stats['source_bytes_m']}, "
                f"F={stats['source_bytes_f']} | "
                f"pre-BER M={pre_ber_m:.5f}, "
                f"F={pre_ber_f:.5f} | "
                f"post-BER M={post_ber_m:.5f}, "
                f"F={post_ber_f:.5f} | "
                f"JP2 M={m_status}, F={f_status} | "
                f"PSNR M={psnr_m:.2f}, "
                f"F={psnr_f:.2f} | "
                f"SSIM M={ssim_m:.4f}, "
                f"F={ssim_f:.4f}"
            )

            # 每張圖片的詳細紀錄
            per_image_results.append({
                "SNR_dB": snr_db,
                "image_index": image_index,

                "MNIST_label": int(label_m),
                "FashionMNIST_label": int(label_f),

                "MNIST_source_bytes": (
                    stats["source_bytes_m"]
                ),
                "FashionMNIST_source_bytes": (
                    stats["source_bytes_f"]
                ),

                "coded_bits_per_stream": (
                    stats["coded_bits_per_stream"]
                ),

                "MNIST_pre_LDPC_BER": pre_ber_m,
                "FashionMNIST_pre_LDPC_BER": pre_ber_f,

                "MNIST_post_LDPC_BER": post_ber_m,
                "FashionMNIST_post_LDPC_BER": post_ber_f,

                "MNIST_JPEG2000_success": success_m,
                "FashionMNIST_JPEG2000_success": success_f,

                "MNIST_PSNR": psnr_m,
                "FashionMNIST_PSNR": psnr_f,

                "MNIST_SSIM": ssim_m,
                "FashionMNIST_SSIM": ssim_f,
            })

            if not success_m:
                print(
                    "    MNIST JPEG2000 error: "
                    f"{stats['jpeg2000_error_m']}"
                )

            if not success_f:
                print(
                    "    Fashion JPEG2000 error: "
                    f"{stats['jpeg2000_error_f']}"
                )

        # ====================================================
        # 儲存該 SNR 的 5×5 圖片
        # ====================================================
        reconstructed_m_batch = torch.stack(
            reconstructed_m_list
        )

        reconstructed_f_batch = torch.stack(
            reconstructed_f_list
        )

        save_image(
            reconstructed_m_batch,
            os.path.join(
                SAVE_ROOT,
                (
                    "recon_JPEG2000_MNIST_"
                    f"5x5_snr{snr_db}.png"
                ),
            ),
            nrow=5,
            padding=2,
            normalize=False,
        )

        save_image(
            reconstructed_f_batch,
            os.path.join(
                SAVE_ROOT,
                (
                    "recon_JPEG2000_FashionMNIST_"
                    f"5x5_snr{snr_db}.png"
                ),
            ),
            nrow=5,
            padding=2,
            normalize=False,
        )

        # ====================================================
        # 計算平均值
        # ====================================================
        mean_pre_ber_m = float(
            np.mean(pre_ber_m_values)
        )

        mean_pre_ber_f = float(
            np.mean(pre_ber_f_values)
        )

        mean_post_ber_m = float(
            np.mean(post_ber_m_values)
        )

        mean_post_ber_f = float(
            np.mean(post_ber_f_values)
        )

        mean_psnr_m = float(
            np.mean(psnr_m_values)
        )

        mean_psnr_f = float(
            np.mean(psnr_f_values)
        )

        mean_ssim_m = float(
            np.mean(ssim_m_values)
        )

        mean_ssim_f = float(
            np.mean(ssim_f_values)
        )

        mnist_success_rate = (
            success_m_count / NUM_IMAGES
        )

        fashion_success_rate = (
            success_f_count / NUM_IMAGES
        )

        mnist_psnr_mean_list.append(mean_psnr_m)
        fashion_psnr_mean_list.append(mean_psnr_f)

        mnist_ssim_mean_list.append(mean_ssim_m)
        fashion_ssim_mean_list.append(mean_ssim_f)

        # 每個 SNR 一列
        snr_results.append({
            "SNR_dB": snr_db,

            "MNIST_PSNR_mean": mean_psnr_m,
            "MNIST_PSNR_max": float(
                np.max(psnr_m_values)
            ),
            "MNIST_PSNR_min": float(
                np.min(psnr_m_values)
            ),

            "MNIST_SSIM_mean": mean_ssim_m,
            "MNIST_SSIM_max": float(
                np.max(ssim_m_values)
            ),
            "MNIST_SSIM_min": float(
                np.min(ssim_m_values)
            ),

            "FashionMNIST_PSNR_mean": mean_psnr_f,
            "FashionMNIST_PSNR_max": float(
                np.max(psnr_f_values)
            ),
            "FashionMNIST_PSNR_min": float(
                np.min(psnr_f_values)
            ),

            "FashionMNIST_SSIM_mean": mean_ssim_f,
            "FashionMNIST_SSIM_max": float(
                np.max(ssim_f_values)
            ),
            "FashionMNIST_SSIM_min": float(
                np.min(ssim_f_values)
            ),

            "MNIST_pre_LDPC_BER_mean": mean_pre_ber_m,
            "FashionMNIST_pre_LDPC_BER_mean": (
                mean_pre_ber_f
            ),

            "MNIST_post_LDPC_BER_mean": mean_post_ber_m,
            "FashionMNIST_post_LDPC_BER_mean": (
                mean_post_ber_f
            ),

            "MNIST_JPEG2000_success_rate": (
                mnist_success_rate
            ),
            "FashionMNIST_JPEG2000_success_rate": (
                fashion_success_rate
            ),

            "MNIST_source_bytes_mean": float(
                np.mean(source_bytes_m_values)
            ),

            "FashionMNIST_source_bytes_mean": float(
                np.mean(source_bytes_f_values)
            ),

            "coded_bits_per_stream_mean": float(
                np.mean(coded_bits_values)
            ),
        })

        print("\nSummary")
        print("-" * 72)

        print(
            f"MNIST JPEG2000 success       : "
            f"{success_m_count}/{NUM_IMAGES} "
            f"({100.0 * mnist_success_rate:.1f}%)"
        )

        print(
            f"Fashion JPEG2000 success     : "
            f"{success_f_count}/{NUM_IMAGES} "
            f"({100.0 * fashion_success_rate:.1f}%)"
        )

        print(
            f"Average pre-LDPC BER MNIST   : "
            f"{mean_pre_ber_m:.6f}"
        )

        print(
            f"Average pre-LDPC BER Fashion : "
            f"{mean_pre_ber_f:.6f}"
        )

        print(
            f"Average post-LDPC BER MNIST  : "
            f"{mean_post_ber_m:.6f}"
        )

        print(
            f"Average post-LDPC BER Fashion: "
            f"{mean_post_ber_f:.6f}"
        )

        print(
            f"Average PSNR MNIST           : "
            f"{mean_psnr_m:.2f} dB"
        )

        print(
            f"Average PSNR Fashion         : "
            f"{mean_psnr_f:.2f} dB"
        )

        print(
            f"Average SSIM MNIST           : "
            f"{mean_ssim_m:.4f}"
        )

        print(
            f"Average SSIM Fashion         : "
            f"{mean_ssim_f:.4f}"
        )

        print("-" * 72)

    # ========================================================
    # CSV：每個 SNR 的平均結果
    # ========================================================
    result_table = pd.DataFrame(snr_results)

    summary_csv_path = os.path.join(
        SAVE_ROOT,
        "metric_results.csv",
    )

    with open(
        summary_csv_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "# Experiment : JPEG2000 + LDPC + QPSK "
            "+ 2x2 MIMO + AWGN + ZF\n"
        )
        file.write(
            "# Dataset    : MNIST + FashionMNIST\n"
        )
        file.write(
            f"# Images     : {NUM_IMAGES} per SNR\n"
        )
        file.write(
            "# Metric     : PSNR / SSIM / BER / "
            "JPEG2000 success rate\n"
        )
        file.write(
            f"# LDPC       : n={LDPC_N}, "
            f"dv={LDPC_DV}, dc={LDPC_DC}, "
            f"maxiter={LDPC_MAX_ITER}\n"
        )
        file.write(
            f"# JPEG2000 irreversible : "
            f"{JPEG2000_IRREVERSIBLE}\n"
        )
        file.write(
            f"# JPEG2000 ratio        : "
            f"{JPEG2000_COMPRESSION_RATIO}\n"
        )
        file.write("\n")

    result_table.to_csv(
        summary_csv_path,
        mode="a",
        index=False,
    )

    # ========================================================
    # CSV：每張圖片的詳細結果
    # ========================================================
    per_image_table = pd.DataFrame(
        per_image_results
    )

    per_image_csv_path = os.path.join(
        SAVE_ROOT,
        "metric_results_per_image.csv",
    )

    per_image_table.to_csv(
        per_image_csv_path,
        index=False,
    )

    # ========================================================
    # PSNR 圖
    # ========================================================
    plt.figure(figsize=(7, 5))

    plt.plot(
        SNR_LIST,
        mnist_psnr_mean_list,
        "o-",
        label="MNIST",
    )

    plt.plot(
        SNR_LIST,
        fashion_psnr_mean_list,
        "s-",
        label="FashionMNIST",
    )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Average PSNR (dB)")
    plt.title("JPEG2000 + LDPC + 2x2 MIMO: PSNR")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    psnr_plot_path = os.path.join(
        SAVE_ROOT,
        "JPEG2000_PSNR_vs_SNR.png",
    )

    plt.savefig(
        psnr_plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    # ========================================================
    # SSIM 圖
    # ========================================================
    plt.figure(figsize=(7, 5))

    plt.plot(
        SNR_LIST,
        mnist_ssim_mean_list,
        "o-",
        label="MNIST",
    )

    plt.plot(
        SNR_LIST,
        fashion_ssim_mean_list,
        "s-",
        label="FashionMNIST",
    )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Average SSIM")
    plt.title("JPEG2000 + LDPC + 2x2 MIMO: SSIM")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    ssim_plot_path = os.path.join(
        SAVE_ROOT,
        "JPEG2000_SSIM_vs_SNR.png",
    )

    plt.savefig(
        ssim_plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    print("\n" + "=" * 72)
    print("Finished.")
    print(f"Results directory : {SAVE_ROOT}")
    print(f"Summary CSV       : {summary_csv_path}")
    print(f"Per-image CSV     : {per_image_csv_path}")
    print(f"PSNR plot         : {psnr_plot_path}")
    print(f"SSIM plot         : {ssim_plot_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
