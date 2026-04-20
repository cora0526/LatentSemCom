import numpy as np
import torch
import time
import socket
import json
import struct

def quantization(z, bits=8, clip_range=(-4.0, 4.0)):
    """
    Uniform quantization with fixed range.

    input:
        z : float tensor
        bits : bit 數
        clip_range : (min, max)

    output:
        z_q   : quantized int tensor
        z_rec : recovered float tensor
        scale : quantization step size
    """

    z_min, z_max = clip_range

    # quantization level
    qmax = 2**(bits - 1) - 1
    qmin = -2**(bits - 1)

    # step size
    scale = (z_max - z_min) / (qmax - qmin)

    # clip
    z_clipped = torch.clamp(z, z_min, z_max)

    # quantize
    z_q = torch.round((z_clipped - z_min) / scale + qmin).clamp(qmin, qmax).to(torch.int8)

    # recover
    z_rec = (z_q.float() - qmin) * scale + z_min

    return z_q, z_rec, scale

def dequantization(z_q, bits=8, clip_range=(-4.0, 4.0)):
    """
    將 quantized int 還原成 float

    input:
        z_q : int tensor (通常是 int8)
        scale : quantization step size
        bits : bit 數（要和 quantization 一致）
        clip_range : (min, max)

    output:
        z_rec : float tensor
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    z_min, z_max = clip_range

    qmax = 2**(bits - 1) - 1
    qmin = -2**(bits - 1)

    scale = (z_max - z_min) / (qmax - qmin)
    
    if isinstance(z_q, np.ndarray):
        z_q = torch.from_numpy(z_q)

    # device 對齊（可選）
    if device is not None:
        z_q = z_q.to(device)

    # 轉 float
    z_q = z_q.float()

    # dequantize
    z_rec = (z_q - qmin) * scale + z_min

    return z_rec



def array_to_bits(x, bit_width=8):
    """
    input:
        x : numpy array 或 torch tensor
        bit_width : 每個數值要保留的 bit 數 (<=8)

    output:
        1D bit array (0/1)
    """

    # tensor → numpy
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    # 轉 uint8
    x = x.astype(np.uint8)

    # unpack 成 8 bits
    bits = np.unpackbits(x[:, None], axis=1, bitorder="little")

    # 只保留前 bit_width 個 bits
    bits = bits[:, :bit_width]

    # 展平
    bits = bits.reshape(-1)

    return bits

def bits_to_array(x_hat, bit_width=8, shape=None, return_tensor=False, dtype=None):
    """
    input:
        x_hat : 1D bit array (0/1)
        bit_width : 每個數值的 bit 數 (<=8)
        shape : 還原後的形狀，例如 z_q.shape
        return_tensor : 是否回傳 torch tensor
        dtype : 回傳 tensor 的型態
                預設:
                  - bit_width == 8 -> torch.int8
                  - bit_width < 8  -> torch.uint8

    output:
        array (numpy 或 torch)
    """

    if isinstance(x_hat, torch.Tensor):
        x_hat = x_hat.detach().cpu().numpy()

    x_hat = np.asarray(x_hat).astype(np.uint8).reshape(-1)

    assert 1 <= bit_width <= 8, "bit_width 必須在 1~8 之間"
    assert len(x_hat) % bit_width == 0, "bit 長度錯誤"

    # reshape 成每個元素對應的 bits
    bits = x_hat.reshape(-1, bit_width)

    # 補成 8 bits
    if bit_width < 8:
        pad = np.zeros((bits.shape[0], 8 - bit_width), dtype=np.uint8)
        bits = np.concatenate([bits, pad], axis=1)

    # little-endian pack 回 uint8
    x_uint8 = np.packbits(bits, axis=1, bitorder="little").reshape(-1)

    # 還原 shape
    if shape is not None:
        x_uint8 = x_uint8.reshape(shape)

    # bit_width=8 時，通常你原本是 int8 量化值，要轉回 signed
    if bit_width == 8:
        x_rec = x_uint8.view(np.int8)
    else:
        x_rec = x_uint8

    if return_tensor:
        if dtype is None:
            dtype = torch.int8 if bit_width == 8 else torch.uint8
        return torch.tensor(x_rec, dtype=dtype)

    return x_rec

def send_to_matlab(data, host="127.0.0.1", port=5000):

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    start = time.perf_counter()

    s.sendall(data.astype(np.float64).tobytes())

    ack = s.recv(3)

    end = time.perf_counter()

    s.close()

    rtt = end - start

    return ack, rtt

def send_msg(sock, obj):
    payload = json.dumps(obj).encode('utf-8')
    header = struct.pack('!I', len(payload))   # 4 bytes unsigned int, big-endian
    sock.sendall(header + payload)

def recv_exact(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Socket closed while receiving data")
        data += packet
    return data

def recv_msg(sock):
    header = recv_exact(sock, 4)
    msg_len = struct.unpack('!I', header)[0]
    payload = recv_exact(sock, msg_len)
    return json.loads(payload.decode('utf-8'))
