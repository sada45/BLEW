import sys
sys.path.append("./scripts")
import numpy as np
import config
import torch
import data_collection.BLong_preamble_detection as pd
import utils
import matplotlib.pyplot as plt
from scramble_table import *

f = None

cur_data = None
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
power_of_2 = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
target_aa_bit = np.array([0,0,0,1,1,0,1,1,0,1,1,1,1,1,0,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1], dtype=np.uint8)
preamble_bits = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
header_bits = np.array([0,0,0,0,1,1,1,0], dtype=np.uint8)
len_bits = np.array([1,1,1,1,1,1,1,1], dtype=np.uint8)

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

def symbol_slice(signal, filtered_signal, noise, preamble_idx, extf, data, extra=config.extra):
    sym_1 = []
    sym_0 = []
    f_sym_1 = []
    f_sym_0 = []
    noise_sym_0 = []
    noise_sym_1 = []
    if extf > 1:
        bits_pre_pdu = 2040 // extf
    else:
        bits_pre_pdu = 265 * 8
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
                if noise is not None:
                    noise_sym_0.append(noise[cur_ptr-extra:et+extra])
            else:
                sym_1.append(signal[cur_ptr-extra:et+extra])
                f_sym_1.append(filtered_signal[cur_ptr-extra:et+extra])
                if noise is not None:
                    noise_sym_1.append(noise[cur_ptr-extra:et+extra])
            cur_ptr = et
    return np.vstack(sym_0), np.vstack(sym_1), np.vstack(f_sym_0), np.vstack(f_sym_1), np.vstack(noise_sym_0), np.vstack(noise_sym_1)

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
    cut_len = num * sym.shape[0]
   
    for i in range(num):
        st = i * config.sample_pre_symbol * target_extf
        et = st + config.sample_pre_symbol * target_extf + 2 * config.extra
        dext_sym.append(sym[:cut_len, st:et])
        if f_sym is not None:
            dext_f_sym.append(f_sym[:cut_len, st:et])
        if noise is not None:
            dext_noise.append(noise[:cut_len, st:et])
    dext_sym = np.vstack(dext_sym)
    if f_sym is not None:
        dext_f_sym = np.vstack(dext_f_sym)
    if noise is not None:
        dext_noise = np.vstack(dext_noise)
    dext_label = np.concatenate([labels[:cut_len] for _ in range(num)])
    if f_sym is None:
        dext_f_sym = None
    if noise is None:
        dext_noise = None
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
    if noise_f_name is not None:
        noise_segs = []
        noise = np.load(noise_f_name)["arr_0"]
        noise = noise.flatten()
    else:
        noise = None
    sym_0, sym_1, f_sym_0, f_sym_1, noise_sym_0, noise_sym_1 = symbol_slice(data[0, :],filtered_signal[0, :], noise, p_idx, extf, np.concatenate([pd.preamble_bits, pd.target_aa_bit]))
    sym = np.vstack([sym_0, sym_1])
    f_sym = np.vstack([f_sym_0, f_sym_1])
    label = np.concatenate([np.zeros(sym_0.shape[0], dtype=np.float32), np.ones(sym_1.shape[0], dtype=np.float32)])

    if noise is not None:
        noise_segs = np.vstack([noise_sym_0, noise_sym_1])
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


def save_dext_slices(syms, f_syms, noises, labels, extf, target_extf, output_name, precent=0.8, cut_len=None):
    if target_extf > extf:
        raise Exception("Too large target extension factor")
    # We don not need many samples 
    syms, f_syms, noises, labels = symbol_dext(syms, f_syms, noises, labels, extf, target_extf)
    shuffle_idx = np.random.permutation(syms.shape[0])
    print(shuffle_idx[:10])
    # We don not need many samples 
    if cut_len is None:
        cut_len = syms.shape[0]
    if cut_len > 2e5:
        print("cut to 2e5")
        cut_len = int(2e5)

    train_num = int(precent * cut_len)

    if precent < 1:
        np.savez("./processed_data/" + output_name + "_train.npz", sym=syms[shuffle_idx[:train_num]], f_sym=(f_syms[shuffle_idx[:train_num]] if f_syms is not None else None), label=labels[shuffle_idx[:train_num]], noise=(noises[shuffle_idx[:train_num]] if noises is not None else None))
        np.savez("./processed_data/" + output_name + "_test.npz", sym=syms[shuffle_idx[train_num:cut_len]], f_sym=(f_syms[shuffle_idx[train_num:cut_len]] if f_syms is not None else None), label=labels[shuffle_idx[train_num:cut_len]], noise=(noises[shuffle_idx[train_num:cut_len]] if noises is not None else None))
        return None, None, None
    else:
        syms = syms[shuffle_idx[:cut_len]]
        f_syms = (f_syms[shuffle_idx[:cut_len]] if f_syms is not None else None)
        noises = (noises[shuffle_idx[:cut_len]] if noises is not None else None)
        labels = labels[shuffle_idx[:cut_len]]
        np.savez("./processed_data/" + output_name + ".npz", sym=syms, f_sym=f_syms, label=labels, noise=noises, shuffle_idx=shuffle_idx)
        return syms, f_syms, noises, labels
    
    
    
    
    
    
# def case_study_gt(ground_truth_file, extf):
#     scenario = "indoor"
#     gt_data = np.load("./raw_data/case_study/" + scenario + "/gt/" + ground_truth_file + ".npz")
#     gt_signal = gt_data["arr_0"].flatten()[np.newaxis, :]
#     gt_signal = utils.filter(gt_signal)
#     gt_time = gt_data["arr_1"][0, 0]
#     p_idx, corr = pd.ground_turth(gt_signal, extf, full=False, time_int=50)
#     p_idx = gt_time + p_idx[0] / config.sample_rate
#     name = ground_truth_file
#     print(name)
#     np.savez("./processed_data/gt/" + scenario + "/" + name + "_gt.npz", p_idx=p_idx, gt_time=gt_time)

# def get_all_ground_truth():
#     ctx = torch.multiprocessing.get_context("spawn")
#     pool = ctx.Pool(6)
#     for d in range(5, 41, 5):
#         for extf in [1, 2, 4, 8, 16, 32, 64]:
#             if extf <= 16:
#                 file_name = "{}m_8c_{}e_{}n_0".format(d, extf, 265)
#             else:
#                 file_name = "{}m_8c_{}e_{}n_0".format(d, extf, 100)
#             # case_study_gt(file_name, extf)
#             pool.apply_async(case_study_gt, args=(file_name, extf))
    
# # case_study/indoor/{}m_8c_{}e_{}n_0
# def case_study_process(raw_file, extf, time_diff_thres=0.01, load=True):
#     if extf <= 16:
#         num = 265
#     else:
#         num = 100
#     if extf > 1:
#         data_bits = np.concatenate([target_aa_bit, header_bits, rand_bits])
#     else:
#         data_bits = np.concatenate([target_aa_bit, header_bits, len_bits, rand_bits])
#     if not load:
#         ground_truth_file = raw_file
#         name = ground_truth_file[ground_truth_file.find("case_study/")+len("case_study/"):]
#         gt_data = np.load("./processed_data/gt/" + name + "_gt.npz")
#         gt_p_time = gt_data["p_idx"] - 0.02906287792933027
#         raw_data = np.load("./raw_data/" + raw_file + ".npz")
#         raw_signal_ori = raw_data["arr_0"].flatten()[np.newaxis, :]
#         time_stamp = raw_data["arr_1"][0, 0]
#         raw_signal = utils.filter(raw_signal_ori)
#         p_idx, corr, pa_corr = pd.preamble_detection_arcphase(raw_signal, extf)
#         p_idx = p_idx[0]
#         # print(p_idx)
#         pa_corr = pa_corr[0]
#         pa_corr = np.max(pa_corr, axis=1)
#         p_time = p_idx / config.sample_rate + time_stamp
#         p_time = p_time[:, np.newaxis]
#         time_diff = np.abs(p_time - gt_p_time)
#         time_diff_neg = p_time - gt_p_time
#         bound_flag = False
#         lost_num = 0
#         total_num = 0
#         processed_p_idx = []
#         # plt.figure(figsize=(12, 6))
#         # et = int(64e4)
#         # a = utils.get_phase_cum_1d(raw_signal[0, :et])
#         # plt.plot(a)
#         # for p in p_idx:
#         #     if p < et:
#         #         plt.plot([p, p], [np.min(a), np.max(a)])
#         # plt.savefig("./figs/test.pdf")
#         # plt.close()
#         for i in range(len(gt_p_time)):
#             # If the data receiving of the raw data is done, we terminate the processing
#             if gt_p_time[i] > p_time[-1]:
#                 break
#             time_diff_seg = time_diff[:, i]
#             print(time_diff_neg[np.argmin(time_diff_seg), i]*1000)
#             # If after the first raw packet, all packets are take into account
#             if bound_flag:
#                 total_num += 1
            
#             # If there is no preamble with the close time with the ground truth
#             avi_idx = np.where(time_diff_seg <= time_diff_thres)[0]
#             if len(avi_idx) == 0:
#                 if bound_flag:
#                     lost_num += 1
#                 continue 
#             elif not bound_flag:
#                 bound_flag = True
#             avi_pa_corr = pa_corr[avi_idx]
#             best_fit_idx = np.argmax(avi_pa_corr)
#             processed_p_idx.append(p_idx[avi_idx[best_fit_idx]])
#         print("Packet lose rate =", lost_num / total_num)
#         # Then we start to slice the data bytes
#         skip_len = (80 + 150) * config.sample_pre_symbol
#         if extf > 1:
#             bits_pre_pdu = int(2040 // extf)
#         else:
#             bits_pre_pdu = 265 * 8
#         packets = []
#         f_packets = []
#         for i in range(len(processed_p_idx)):
#             pkt = []
#             f_pkt = []
#             ptr = 8 * extf * config.sample_pre_symbol + processed_p_idx[i]
#             for j in range(8, num*8):
#                 if j % bits_pre_pdu == 0:
#                     ptr += skip_len
#                 n_ptr = ptr + extf * config.sample_pre_symbol
#                 sp = ptr - config.extra
#                 ep = n_ptr + config.extra
#                 pkt.append(raw_signal_ori[0, sp:ep])
#                 f_pkt.append(raw_signal[0, sp:ep])
#                 ptr = n_ptr
#             packets.append(pkt)
#             f_packets.append(f_pkt)
#         packets = np.array(packets)
#         f_packets = np.array(f_packets)
#         np.savez("./processed_data/" + raw_file + ".npz", f_packets=f_packets, packets=packets, lost_num=lost_num, total_num=total_num)
#     else:
#         data = np.load("./processed_data/" + raw_file + ".npz")
#         # f_packets = data["f_packets"]
#         packets = data["packets"]
#         lost_num = data["lost_num"]
#         total_num = data["total_num"]
#     packets_shape = packets.shape
#     packets = packets.reshape([-1, packets.shape[2]])
#     packets = utils.signal_normalize(packets, 1)
#     f_packets = utils.filter(packets).astype(np.complex64)
#     packets = packets.reshape(packets_shape)
#     f_packets = f_packets.reshape(packets_shape)
#     packets = packets[:, :, config.extra:packets.shape[2]-config.extra]
#     f_packets = f_packets[:, :, config.extra:f_packets.shape[2]-config.extra]
#     cpns, bcrs = dnn.dnn_decode(packets, f_packets, extf, data_bits[:(num-1)*8])
#     total_bits = packets.shape[0] * packets.shape[1]
#     bcrs = bcrs / total_bits
#     cpns = cpns / packets.shape[0]
#     print(raw_file, extf, "bcr:", bcrs, "cpns:", cpns)

def wifi_int_exp(dextf=8):
    extf = 64
    raw_signal = np.load("/data/liym/BLong/raw_data/case_study/gt/0m_8c_64e_100n_2.npz")["arr_0"].flatten()[np.newaxis, :]
    noise = np.load("/data/liym/BLong/raw_data/case_study/wifi/5m_4c_40M_0.npz")["arr_0"].flatten()[np.newaxis, :]
    cut_len = np.min([raw_signal.shape[1], noise.shape[1]])
    raw_signal = raw_signal[:, :cut_len]
    noise = noise[:, :cut_len]
    
    filtered_signal = utils.filter(raw_signal)
    p_idx, _ = pd.ground_turth(raw_signal, extf, False)
    data_bits = np.concatenate([preamble_bits, target_aa_bit, header_bits, rand_bits])
    for snr in range(-20, 1, 2):
        int_signal = utils.wifi_awgn(filtered_signal[0, :], snr, extf, p_idx[0], noise[0, :])
        sym_0, sym_1, _, _, _, _ = symbol_slice(int_signal, filtered_signal[0, :], noise[0, :], p_idx[0], extf, data_bits[:100], extra=config.extra)
        syms = np.vstack([sym_0, sym_1])
        labels = np.concatenate([np.zeros(sym_0.shape[0], dtype=np.uint8), np.ones(sym_1.shape[0], dtype=np.uint8)])
        bits, correct_num = dnn.decode_ber(syms, labels, extf)
        ber = (syms.shape[0] - correct_num) / syms.shape[0]
        print("snr={}, extf={}, dnn={}, stft={}, pd={},".format(snr, extf, ber[0], ber[1], ber[2]), end=",")
        dext_syms, _, _, dext_labels = symbol_dext(syms, None, None, labels, extf, dextf)
        dext_bits, dext_correct_num = dnn.decode_ber(dext_syms, dext_labels, dextf)
        ber = (dext_syms.shape[0] - dext_correct_num) / dext_syms.shape[0]
        print("dextf={}, dext_dnn={}, dext_stft={}, dext_pd={}".format(snr, extf, ber[0], ber[1], ber[2]))

# def time_diff_cmp():
#     gd1 = np.load("./processed_data/gt/0m_8c_64e_100n_2_gt.npz")
#     gd2 = np.load("./processed_data/gt/0m_8c_64e_100n_3_gt.npz")
#     gt1 = gd1["p_idx"][:, np.newaxis]
#     gt2 = gd2["p_idx"]
#     d = gt2 - gt1
#     min_idx = np.argmin(np.abs(d), axis=1)
#     min_time_diff = d[(np.arange(d.shape[0]), min_idx)]
#     min_time_diff = min_time_diff[np.where((min_time_diff<0.03)&(min_time_diff>0.02))]
#     print(np.mean(min_time_diff))
    
# if __name__ == "__main__":
#     get_all_ground_truth()
#     case_study_gt("case_study/indoor/gt/5m_8c_1e_265n_0", 1)
#     for d in range(5, 41, 5):
#         for extf in utils.power_of_2:
#             if extf > 16:
#                 case_study_process("case_study/indoor/{}m_8c_{}e_{}n_0".format(d, extf, 100), extf, load=False)
#             else:
#                 case_study_process("case_study/indoor/{}m_8c_{}e_{}n_0".format(d, extf, 265), extf, load=False)
   
   
   
   
   
   
   
    # get_all_ground_truth()
    # case_study_gt("case_study/gt/0m_8c_64e_100n_2", 64)
    # case_study_gt("case_study/gt/0m_8c_64e_100n_3", 64)
    # time_diff_cmp()
    
    