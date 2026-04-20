from gnuradio import gr, blocks, analog
import numpy as np

class MISO_Channel(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self)

        # ====== 兩個 transmitter ======
        self.x1 = np.array([1, 2, 3, 4], dtype=np.float32)
        self.x2 = np.array([0.5, -1, 2, 1], dtype=np.float32)

        src1 = blocks.vector_source_f(self.x1.tolist(), repeat=False)
        src2 = blocks.vector_source_f(self.x2.tolist(), repeat=False)

        # ====== channel ======
        self.h1 = 0.8
        self.h2 = 0.3

        mul1 = blocks.multiply_const_ff(self.h1)
        mul2 = blocks.multiply_const_ff(self.h2)

        # ====== 相加 ======
        adder = blocks.add_ff()

        # ====== noise ======
        self.noise_vec = np.array([0.01, -0.02, 0.03, 0.00], dtype=np.float32)
        noise_src = blocks.vector_source_f(self.noise_vec.tolist(), repeat=False)

        noise_adder = blocks.add_ff()

        # ====== output ======
        dst = blocks.vector_sink_f()

        # ====== connect ======
        self.connect(src1, mul1)
        self.connect(src2, mul2)

        self.connect(mul1, (adder, 0))
        self.connect(mul2, (adder, 1))

        self.connect(adder, (noise_adder, 0))
        self.connect(noise_src, (noise_adder, 1))

        self.connect(noise_adder, dst)

        self.dst = dst


if __name__ == "__main__":
    tb = MISO_Channel()
    tb.run()

    print("x1 =", tb.x1)
    print("x2 =", tb.x2)
    print("noise =", tb.noise_vec)
    print("Received y:")
    print(tb.dst.data())