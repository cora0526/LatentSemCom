import numpy as np
import torch
import time
import socket

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
