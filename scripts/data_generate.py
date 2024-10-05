import sys
sys.path.append("./scripts")
import numpy as np
# import matplotlib.pyplot as plt
import config
import torch
import torch.nn.functional as F
import scipy
import time
import data_collection.BLong_preamble_detection as pd
import utils
from concurrent.futures import ProcessPoolExecutor 

cur_data = None
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
power_of_2 = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)

# def get_stft_symbols(raw_symbols, extf, win_size=config.stft_window_size, cpu=True):
#     if isinstance(raw_symbols, np.ndarray):
#         raw_symbols = torch.from_numpy(raw_symbols)
#     raw_symbols = raw_symbols.to(device)
#     if len(raw_symbols.shape) == 1:
#         raw_symbols = raw_symbols.unsqueeze(0)
#     # We pad some 0s to make sure the last few sample can be considered into the STFT
#     if win_size > 1:
#         raw_padding = torch.zeros((int(raw_symbols.shape[0]), (win_size-1)*config.sample_pre_symbol), dtype=torch.complex64).to(device)
#         raw_symbols = torch.cat((raw_symbols, raw_padding), dim=1)
#     # Get the STFT result with given window size and hann window
#     if extf > win_size:
#         raw_symbols = raw_symbols.unfold(1, config.sample_pre_symbol * win_size, config.sample_pre_symbol)
#     else:
#         # If the extended symbol is shorter than STFT window, we need to pad more zeros
#         # Since we have the hanning window, it would be better to pad zeros on the two side (left and right)
#         raw_symbols = raw_symbols.unfold(1, config.sample_pre_symbol * extf, config.sample_pre_symbol)
#         left_pads_num = (win_size * config.sample_pre_symbol - extf * config.sample_pre_symbol) // 2
#         right_pads_num =  win_size * config.sample_pre_symbol - extf * config.sample_pre_symbol - left_pads_num
#         pads_left = torch.zeros((raw_symbols.shape[0], raw_symbols.shape[1], left_pads_num), dtype=torch.complex64).to(device)
#         pads_right = torch.zeros((raw_symbols.shape[0], raw_symbols.shape[1], right_pads_num), dtype=torch.complex64).to(device)
#         raw_symbols = torch.cat((pads_left, raw_symbols, pads_right), dim=2)

#     # Adding the hanning window
#     raw_symbols = raw_symbols * torch.hann_window(config.sample_pre_symbol * win_size).to(device)
#     # Remove the DC value
#     raw_symbols_mean = torch.mean(raw_symbols, dim=2).unsqueeze(2)
#     raw_symbols = raw_symbols - raw_symbols_mean
#     stft_res = torch.fft.fft(raw_symbols, n=config.sample_pre_symbol*win_size, dim=2).transpose(1, 2).contiguous()  # [B, F, T]
#     stft_res = torch.stack((stft_res.real, stft_res.imag), dim=1)  # [B, 2, F, T]
#     # Since we have LPF to filter out the high-frequency noise, the middle part of each FFT frame can be removed.
#     # just remain the 2M bandwidth, 1M for high freq, and 1M for low freq
#     # freq_resolution = 1 / (config.stft_window_size * 1e-6)
#     # config.sample_rate / (win_size * config.sample_pre_symbol)
#     rem_num = config.stft_window_size
#     lpf_stft_shape = (int(stft_res.shape[0]), int(stft_res.shape[1]), 2 * rem_num, int(stft_res.shape[3]))
#     lpf_stft_res = torch.zeros(lpf_stft_shape, dtype=torch.float32).to(device)
#     lpf_stft_res[:, :, 0:rem_num, :] = stft_res[:, :, 0:rem_num, :]
#     lpf_stft_res[:, :, rem_num:, :] = stft_res[:, :, (stft_res.shape[2]-rem_num):, :]
#     if cpu:
#         return lpf_stft_res.cpu()
#     else:
#         return lpf_stft_res

def get_stft_symbols(raw_symbols, extf, win_size=config.stft_window_size, cpu=True):
    if isinstance(raw_symbols, np.ndarray):
        raw_symbols = torch.from_numpy(raw_symbols)
    raw_symbols = raw_symbols.to(device)
    if len(raw_symbols.shape) == 1:
        raw_symbols = raw_symbols.unsqueeze(0)
    # We pad some 0s to make sure the last few sample can be considered into the STFT

    # Get the STFT result with given window size and hann window
    if extf > win_size:
        if win_size > 1:
            raw_padding = torch.zeros((int(raw_symbols.shape[0]), (win_size-1)*config.sample_pre_symbol), dtype=torch.complex64).to(device)
            raw_symbols = torch.cat((raw_symbols, raw_padding), dim=1)
        raw_symbols = raw_symbols.unfold(1, config.sample_pre_symbol * win_size, config.sample_pre_symbol)
    else:
        # If the extended symbol is shorter than STFT window, we need to pad more zeros
        # Since we have the hanning window, it would be better to pad zeros on the two side (left and right)
        if extf >= 2:
            raw_padding = torch.zeros((int(raw_symbols.shape[0]), (extf-1)*config.sample_pre_symbol), dtype=torch.complex64).to(device)
            raw_symbols = torch.cat((raw_symbols, raw_padding), dim=1)
        raw_symbols = raw_symbols.unfold(1, config.sample_pre_symbol * extf, config.sample_pre_symbol)
        left_pads_num = (win_size * config.sample_pre_symbol - extf * config.sample_pre_symbol) // 2
        right_pads_num =  win_size * config.sample_pre_symbol - extf * config.sample_pre_symbol - left_pads_num
        pads_left = torch.zeros((raw_symbols.shape[0], raw_symbols.shape[1], left_pads_num), dtype=torch.complex64).to(device)
        pads_right = torch.zeros((raw_symbols.shape[0], raw_symbols.shape[1], right_pads_num), dtype=torch.complex64).to(device)
        raw_symbols = torch.cat((pads_left, raw_symbols, pads_right), dim=2)

    # Adding the hanning window
    raw_symbols = raw_symbols * torch.hann_window(config.sample_pre_symbol * win_size).to(device)
    # Remove the DC value
    raw_symbols_mean = torch.mean(raw_symbols, dim=2).unsqueeze(2)
    raw_symbols = raw_symbols - raw_symbols_mean
    stft_res = torch.fft.fft(raw_symbols, n=config.sample_pre_symbol*win_size, dim=2).transpose(1, 2).contiguous()  # [B, F, T]
    stft_res = torch.stack((stft_res.real, stft_res.imag), dim=1)  # [B, 2, F, T]
    # Since we have LPF to filter out the high-frequency noise, the middle part of each FFT frame can be removed.
    # just remain the 2M bandwidth, 1M for high freq, and 1M for low freq
    # freq_resolution = 1 / (config.stft_window_size * 1e-6)
    # config.sample_rate / (win_size * config.sample_pre_symbol)
    rem_num = config.stft_window_size
    lpf_stft_shape = (int(stft_res.shape[0]), int(stft_res.shape[1]), 2 * rem_num, int(stft_res.shape[3]))
    lpf_stft_res = torch.zeros(lpf_stft_shape, dtype=torch.float32).to(device)
    lpf_stft_res[:, :, 0:rem_num, :] = stft_res[:, :, 0:rem_num, :]
    lpf_stft_res[:, :, rem_num:, :] = stft_res[:, :, (stft_res.shape[2]-rem_num):, :]
    if cpu:
        return lpf_stft_res.cpu()
    else:
        return lpf_stft_res

def symbol_slice(signal, filtered_signal, preamble_idx, extf, data, extra=config.extra):
    sym_1 = []
    sym_0 = []
    f_sym_1 = []
    f_sym_0 = []
    bits_pre_pdu = 2040 // extf
    symbol_len = extf * config.sample_pre_symbol

    for i in range(len(preamble_idx)):
        cur_ptr = preamble_idx[i]
        for j in range(len(data)):
            if j > 0 and j % bits_pre_pdu == 0:
                cur_ptr += (150 + 80) * config.sample_pre_symbol
            
            et = cur_ptr + symbol_len
            if data[j] == 0:
                sym_0.append(signal[cur_ptr-extra:et+extra])
                f_sym_0.append(filtered_signal[cur_ptr-extra:et+extra])
            else:
                sym_1.append(signal[cur_ptr-extra:et+extra])
                f_sym_1.append(filtered_signal[cur_ptr-extra:et+extra])
            cur_ptr = et
    return np.vstack(sym_0), np.vstack(sym_1), np.vstack(f_sym_0), np.vstack(f_sym_1)

def symbol_dext(sym, f_sym, noise, labels, extf, target_extf):
    if target_extf >= extf:
        return sym, f_sym, noise, labels
    if extf % target_extf != 0:
        raise Exception("extf % target_extf != 0")
    num = int(extf // target_extf)
    valid_len = num * target_extf * config.sample_pre_symbol
    dext_sym = []
    dext_f_sym = []
    dext_noise = []
    dext_label = []
    for i in range(num):
        st = i * config.sample_pre_symbol * target_extf
        et = st + config.sample_pre_symbol * target_extf + 2 * config.extra
        dext_sym.append(sym[:, st:et])
        dext_f_sym.append(f_sym[:, st:et])
        dext_noise.append(noise[:, st:et])
    dext_sym = np.vstack(dext_sym)
    dext_f_sym = np.vstack(dext_f_sym)
    dext_noise = np.vstack(dext_noise)
    dext_label = np.concatenate([labels for _ in range(num)])
    return dext_sym, dext_f_sym, dext_noise, dext_label

def data_preprocessing(f_name, noise_f_name, extf, synthesis):
    data = np.load(f_name)
    raw_signal = data["arr_0"]
    time_stamp = data["arr_1"]
    data = raw_signal.flatten()[np.newaxis, :]
    print("start")
    filtered_signal = utils.filter(data).astype(np.complex64)
    if synthesis:
        p_idx, _ = pd.ground_turth(filtered_signal, extf, full=False)
    else:
        p_idx, corr, pa_corr = pd.preamble_detection(filtered_signal, extf, cluster_size=3, full=True)
    p_idx = p_idx[0]
    sym_0, sym_1, f_sym_0, f_sym_1 = symbol_slice(data[0, :],filtered_signal[0, :], p_idx, extf, np.concatenate([pd.preamble_bits, pd.target_aa_bit]))
    sym = np.vstack([sym_0, sym_1])
    f_sym = np.vstack([f_sym_0, f_sym_1])
    label = np.concatenate([np.zeros(sym_0.shape[0], dtype=np.float32), np.ones(sym_1.shape[0], dtype=np.float32)])

    if noise_f_name is not None:
        noise_segs = []
        noise = np.load(noise_f_name)["arr_0"]
        noise = noise.flatten()
        sp = 0
        move_len = sym.shape[1] - config.extra
        for i in range(sym.shape[0]):
            ep = sp + sym.shape[1]
            if ep >= len(sym):
                sp = 0
                ep = sym.shape[1]
            noise_seg = noise[sp:ep]
            noise_segs.append(noise_seg)
            sp += move_len
        noise_segs = np.vstack(noise_segs)
    else:
        noise_segs = np.zeros(sym.shape, dtype=np.complex64)
        noise_segs.real = np.random.randn(sym.shape[0], sym.shape[1])
        noise_segs.imag = np.random.randn(sym.shape[0], sym.shape[1])
    return sym, f_sym, label, noise_segs

def process_and_save(signal_prefix, noise_prefix, batch_num, extf, output_name, synthesis, precent=0.8):
    syms = []
    f_syms = []
    labels = []
    noises = []

    for b in batch_num:
        signal_f_name = "./raw_data/" + signal_prefix + str(b) + ".npz"
        if noise_prefix is not None:
            noise_f_name = "./raw_data/" + noise_prefix + str(b) + ".npz"
        else:
            noise_f_name = None
        sym, f_sym, label, noise = data_preprocessing(signal_f_name, noise_f_name, extf, synthesis)
        syms.append(sym)
        f_syms.append(f_sym)
        labels.append(label)
        noises.append(noise)

    syms = np.vstack(syms)
    labels = np.concatenate(labels)
    noises = np.vstack(noises)
    f_syms = np.vstack(f_syms)
    shuffle_idx = np.random.permutation(syms.shape[0])
    train_num = int(precent * syms.shape[0])
    if precent < 1:
        np.savez("./processed_data/" + output_name + "_train.npz", sym=syms[shuffle_idx[:train_num]], f_sym=f_syms[shuffle_idx[:train_num]], label=labels[shuffle_idx[:train_num]], noise=noises[shuffle_idx[:train_num]])
        np.savez("./processed_data/" + output_name + "_test.npz", sym=syms[shuffle_idx[train_num:]], f_sym=f_syms[shuffle_idx[train_num:]], label=labels[shuffle_idx[train_num:]], noise=noises[shuffle_idx[train_num:]])
    else:
        np.savez("./processed_data/" + output_name + ".npz", sym=syms, f_sym=f_syms, label=labels, noise=noises)
    return syms, f_syms, noises, labels


def save_dext_slices(syms, f_syms, noises, labels, extf, target_extf, output_name, precent=0.8):
    if target_extf > extf:
        raise Exception("Too large target extension factor")
    # We don not need many samples 
    cut_len = 2 * syms.shape[0]
    syms, f_syms, noises, labels = symbol_dext(syms, f_syms, noises, labels, extf, target_extf)
    shuffle_idx = np.random.permutation(syms.shape[0])
    # We don not need many samples 

    train_num = int(precent * cut_len)

    if precent < 1:
        np.savez("./processed_data/" + output_name + "_train.npz", sym=syms[shuffle_idx[:train_num]], f_sym=f_syms[shuffle_idx[:train_num]], label=labels[shuffle_idx[:train_num]], noise=noises[shuffle_idx[:train_num]])
        np.savez("./processed_data/" + output_name + "_test.npz", sym=syms[shuffle_idx[train_num:cut_len]], f_sym=f_syms[shuffle_idx[train_num:cut_len]], label=labels[shuffle_idx[train_num:cut_len]], noise=noises[shuffle_idx[train_num:cut_len]])
    else:
        np.savez("./processed_data/" + output_name + ".npz", sym=syms[shuffle_idx[:cut_len]], f_sym=f_syms[shuffle_idx[:cut_len]], label=labels[shuffle_idx[:cut_len]], noise=noises[shuffle_idx[:cut_len]])

if __name__ == "__main__":
    origin_extf = 64
    save_prefix = "white_noise_new_{}e"

    syms, f_syms, noises, labels = process_and_save("case_study/indoor/0m_4c_64e_5n_", None, batch_num=[0], extf=origin_extf, output_name=save_prefix.format(origin_extf), synthesis=True)
    extfs = [1, 2, 4, 8, 16, 32]
    with ProcessPoolExecutor(6) as p:
        for target_extf in extfs:
            p.submit(save_dext_slices, syms, f_syms, noises, labels, origin_extf, target_extf, save_prefix.format(target_extf))
        p.shutdown()
