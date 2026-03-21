import socket
import numpy as np
import time

HOST = "127.0.0.1"
PORT = 5000

data = np.random.randn(64).astype(np.float64)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

start = time.perf_counter()
s.sendall(data.tobytes())

ack = s.recv(3)   # 一定要先等 ACK
end = time.perf_counter()

print("ACK:", ack)
print("RTT:", end - start)

s.close()
