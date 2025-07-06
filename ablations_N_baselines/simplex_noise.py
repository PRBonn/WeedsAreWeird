"""A small test script to get simplex noise
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

import sys
sys.path.append("..//waa_baselines/AnoDDPM")
from simplex import Simplex_CLASS
from GaussianDiffusion import generate_simplex_noise

if __name__ == '__main__':

    noise_shape = (1, 1, 512, 512)

    start = time.time()

    simple_simplex = Simplex_CLASS()
    declaration_time = time.time()
    print("declaration time: ", declaration_time-start)
    x_t = torch.zeros(noise_shape)
    t_distance = 250
    t_tensor = torch.tensor([t_distance], device=x_t.device).repeat(x_t.shape[0])
    start_gen = time.time()
    noise = generate_simplex_noise(
                simple_simplex, 
                x_t, 
                t_tensor,
                # False, 
                # frequency=64, 
                # octave=6,
                # persistence=0.8
            ).float()
    stop_gen = time.time()
    print("gen_time: ", stop_gen - start_gen)

    vis_noise = torch.transpose(noise[0], 0, -1)

    stop = time.time()
    print("total time: ", stop-start)

    plt.matshow(vis_noise)
    plt.colorbar()
    plt.show()

    import pdb; pdb.set_trace()


