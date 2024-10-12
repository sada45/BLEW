import numpy as np
# import matplotlib.pyplot as plt
# We modified the pwlf lib
import sys
sys.path.append("./scripts/")
import time
from opt.ber_relations import *

plf = None
plf_list = []
plf_betas = None
plf_breaks = None
extf_list = [1, 2, 4, 8, 16, 32, 64, 128]
n_max_list = [n for n in range(8, 249, 8)] + [255]
extf_num = 8
N_f = 10
blong_bers = None
dnn_bers = None
stft_bers = None
ber_bound = []

# The code for experiments in real world
def goodput_optimiation_nlos_fixn(bcrs, pdetrs, n_max=255, pdr_thres=None):
    extf_list = np.array([1, 2, 4, 8, 16, 32, 64])
    extf_num = len(extf_list)
    pdetrs = pdetrs
    bcrs = bcrs[np.newaxis, :]
    goodput = np.zeros(extf_num, dtype=np.float64)
    # For extension factor = 1, we reuse the native fields in the native AIPs
    goodput[0] = n_max / (n_max + N_f + 150 / 8) * bcrs[:, 0:1]
    
    # Otherwise, N_f * n_ext extra length needs to be transmitted
    n_total = (n_max + N_f) * extf_list[1:]
    n_bit = np.floor(255 * 8 / extf_list[1:])
    n_aip = np.ceil(n_total * 8 / n_bit)
    goodput[1:] = (n_max * bcrs[:, 1:]) / (n_total + (N_f + 150 / 8) * n_aip)
    goodput = goodput * pdetrs[:]
    if pdr_thres is not None:
        max_idxs = np.argsort(-goodput)
        for i in max_idxs:
            if pdetrs[i] >= pdr_thres:
                return n_max, extf_list[i], goodput[i]
    else:
        max_idx = np.argmax(goodput)
        return n_max, extf_list[max_idx], goodput[max_idx]
    return None, None, None


def goodput_optimiation_snr_fixn(bcrs, pdetrs, n_max=255):
    extf_list = np.array([1, 2, 4, 8, 16, 32, 64])
    extf_num = len(extf_list)
    goodput = np.zeros(extf_num, dtype=np.float64)
    # For extension factor = 1, we reuse the native fields in the native AIPs
    goodput[0] = n_max / (n_max + N_f + 150 / 8) * bcrs[0]
    # Otherwise, N_f * n_ext extra length needs to be transmitted
    n_total = (n_max + N_f) * extf_list[1:]
    n_bit = np.floor(255 * 8 / extf_list[1:])
    n_aip = np.ceil(n_total * 8 / n_bit)
    goodput[1:] = (n_max * bcrs[1:]) / (n_total + (N_f + 150 / 8) * n_aip)
    goodput = goodput * pdetrs[:]
    max_idx = np.argmax(goodput)
    return n_max, extf_list[max_idx], goodput[max_idx]
