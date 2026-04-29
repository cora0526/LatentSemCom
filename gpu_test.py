import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import socket
from com_utils import array_to_bits,quantization,dequantization,bits_to_array, send_to_matlab, send_msg, recv_exact, recv_msg
from mlset_util import import_all_model

device = "cuda" 
G_path = "./save_model/G_wgangp_C.pth"
E_path = "./save_model/E_invert_to_G_C_only.pth"
E_f_path = "./save_model/E_invert_to_G_F.pth"
G,E,E_f = import_all_model()


G.to(device).eval()
E.to(device).eval()