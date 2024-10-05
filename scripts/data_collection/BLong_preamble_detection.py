import sys
sys.path.append("./scripts/")
import numpy as np
import config
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy
import utils
import time
import cv2
from concurrent.futures import ProcessPoolExecutor
from scramble_table import * 
import DANF.blong_dnn as dnn

preamble_bits = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
target_aa_bit = np.array([0,0,0,1,1,0,1,1,0,1,1,1,1,1,0,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1], dtype=np.uint8)
pa_bits = np.array([0,1,0,1,0,1,0,1,0,0,0,1,1,0,1,1,0,1,1,1,1,1,0,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1], dtype=np.uint8)
plt_flag = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
process_pool = ProcessPoolExecutor(12)
# device = torch.device("cpu")

def search_sequence(bits, target_bits):
    res = cv2.matchTemplate(bits, target_bits, cv2.TM_SQDIFF)
    idx = np.where(res==0)[0]
    if len(idx) > 0:
        return idx
    else:
        return None

def raw_signal_segment(data, timestamp, extf):
    i0 = np.real(data[:-1])
    q0 = np.imag(data[:-1])
    i1 = np.real(data[1:])
    q1 = np.imag(data[1:])
    phase_diff = i0 * q1 - i1 * q0
    phase_len = len(phase_diff)
    bits = np.zeros(int(np.ceil(phase_len / config.sample_pre_symbol)), dtype=np.uint8)
    segment_len = extf * 8 + 18 + 32 + 24 + 1 + 16 * extf + 20
    raw_segs = []
    time_stamps = []
    for i in range(config.sample_pre_symbol):
        bits[:] = 0
        sample_len = (phase_len - i) // config.sample_pre_symbol
        p = phase_diff[i: i+sample_len*config.sample_pre_symbol].reshape(-1, config.sample_pre_symbol)
        vote = np.sum(p, 1)
        # vote = phase_diff[i: i+sample_len*config.sample_pre_symbol: config.sample_pre_symbol]
        bits[np.where(vote>0)[0]] = 1
        preamble_idx = search_sequence(bits, target_aa_bit)
        if preamble_idx is not None:
            for j in range(len(preamble_idx)):
                idx = (preamble_idx[j] - 20 - 8 * extf) * config.sample_pre_symbol
                end_idx = idx + config.sample_pre_symbol * segment_len + config.sample_pre_symbol
                if idx >= 0 and end_idx < len(data):
                    raw_data = data[idx: end_idx]
                    time = timestamp[0] + idx * (1 / config.sample_rate)
                    if len(time_stamps) > 0:
                        min_time_diff = np.min(time - np.array(time_stamps))
                        if min_time_diff < 1e-5:
                            continue
                    raw_segs.append(raw_data.copy())
                    time_stamps.append(time)
                else:
                    print("sample too short", i, idx, end_idx)
    return raw_segs, time_stamps

def cal_phase_diff(signal):
    return signal[:, 0:-1].real * signal[:, 1:].imag - signal[:, 1:].real * signal[:, 0:-1].imag

def cal_phase_diff_arc(signal):
    phase = utils.get_phase(signal)
    return phase[:, 1:] - phase[:, 0:-1]

# For low-frequency, the corresponding mask is -1, and 1 for high-frequency
def get_preamble_mask(extf, sample_pre_symbol=config.sample_pre_symbol):
    mask = torch.ones(extf * 8 * sample_pre_symbol, dtype=torch.float32)
    for i in range(0, 8, 2):
        mask[i*extf*sample_pre_symbol: (i+1)*extf*sample_pre_symbol] = -1
    return mask.view((1, 1, -1)).contiguous().to(device)

def get_preamble_aa_mask(extf, full=False):
    bits_pre_pdu = int(2040 // (extf))

    if full:
        zero_pad_len = 2040 - extf * bits_pre_pdu
        interval_mask = torch.zeros(config.sample_pre_symbol * (80 + 150 + zero_pad_len), dtype=torch.float32)
    else:
        interval_mask = torch.zeros(config.sample_pre_symbol * (80 + 150), dtype=torch.float32)
    
    preamble_mask = torch.ones(8 * config.sample_pre_symbol * extf, dtype=torch.float32)
    for i in range(0, 8, 2):
        preamble_mask[i*config.sample_pre_symbol*extf: (i+1)*config.sample_pre_symbol*extf] = -1
    mask = [preamble_mask]
    bits_pre_pdu = int(2040 // (extf))
    bits_num = 8
    bit_mask = torch.ones(config.sample_pre_symbol * extf, dtype=torch.float32)
    for i in range(len(target_aa_bit)):
        if bits_num % bits_pre_pdu == 0:
            mask.append(interval_mask)
        if target_aa_bit[i] == 1:
            mask.append(bit_mask)
        else:
            mask.append(-bit_mask)
        bits_num += 1
    mask = torch.concatenate(mask)
    return mask.view((1, 1, -1)).contiguous().to(device)

def cal_corrlation_normal(phase_diff, mask):
    np_flag = False
    if isinstance(phase_diff, np.ndarray):
        phase_diff_tensor = torch.from_numpy(phase_diff).to(device)
        phase_diff_tensor = phase_diff_tensor.unsqueeze(1).type(torch.float32).to(device)
        np_flag = True
    else:
        phase_diff_tensor = phase_diff.to(device)
        phase_diff_tensor = phase_diff_tensor.unsqueeze(1).type(torch.float32).to(device)
    
    out = F.conv1d(phase_diff_tensor, mask).squeeze(1)
    if np_flag:
        return out.cpu().numpy()
    else:
        return out.cpu()

def cal_corrlation(phase_diff, mask, acc_batch=64):
    np_flag = False
    if isinstance(phase_diff, np.ndarray):
        phase_diff_tensor = torch.from_numpy(phase_diff).to(device)
        phase_diff_tensor = phase_diff_tensor.type(torch.float32).to(device)
        np_flag = True
    else:
        phase_diff_tensor = phase_diff.to(device)
        phase_diff_tensor = phase_diff_tensor.type(torch.float32).to(device)
    
    # For those 1-D long signal, we split them into multiple batches to speed-up
    oned_flag = False 
    if phase_diff_tensor.shape[0] == 1:
        oned_flag = True
        batch_len = phase_diff_tensor.shape[1] // acc_batch
        pad = torch.zeros([1, mask.shape[2]-1], dtype=torch.float32, device=device)
        phase_diff_tensor = torch.hstack([phase_diff_tensor, pad]).to(device)
        phase_diff_tensor = phase_diff_tensor.unfold(1, batch_len+mask.shape[2]-1, batch_len).squeeze(0)
    phase_diff_tensor = phase_diff_tensor.unsqueeze(1)
    out = F.conv1d(phase_diff_tensor, mask).squeeze(1)
    if oned_flag:
        out = out.view(1, -1).contiguous()
    if np_flag:
        return out.cpu().numpy()
    else:
        return out.cpu()

def peak_clustering_preamble_detection_1d(corrlation, extf, cluster_size=3, offset=0, thres_percent=95):
    max_diff = extf * config.sample_pre_symbol * 8
    peaks, properties = scipy.signal.find_peaks(corrlation, prominence=0, height=0)
    promin = properties["prominences"]
    promin_thres = np.percentile(promin, thres_percent)
    # promin_thres = promin_precentile
    height = properties["peak_heights"]
    height_thres = np.percentile(height, thres_percent)
    peaks = peaks[np.where((promin>promin_thres)&(height>height_thres))]

    if len(peaks) > 0:
        peak_clusters = [[peaks[0]]]
        peak_cluster_center = [peaks[0]] # should be the peak that with maximum value
        for j in range(1, len(peaks)):
            # The peak can be clustered into the last cluster
            if peaks[j] - peak_cluster_center[-1] < max_diff:
                # This peak is higher than the original center
                # We consider it as the new center
                if corrlation[peaks[j]] > corrlation[peak_cluster_center[-1]]:
                    peak_cluster_center[-1] = peaks[j]
                # Otherwise, we juse append the peak into the cluster
                peak_clusters[-1].append(peaks[j])
            # This peak is far way from the former cluster
            # we juse take it as a new cluster 
            else:
                # If last cluster is too small, we can consider it as a noise
                # Another exceptation is that the center is on one-side
                # (cluster_size > 1 and (peak_cluster_center[-1] == peak_clusters[-1][0] or \
                #     peak_cluster_center[-1] == peak_clusters[-1][-1]))
                if len(peak_clusters[-1]) < cluster_size:
                    peak_clusters.pop()
                    peak_cluster_center.pop()
                peak_clusters.append([peaks[j]])
                peak_cluster_center.append(peaks[j])
    else:
        peak_clusters = []
        peak_cluster_center = [-1]
    
    # Check wether the last cluster is noise or not
    # (len(peak_clusters[-1]) < cluster_size or \
    #     (cluster_size > 1 and (peak_cluster_center[-1] == peak_clusters[-1][0] or \
    #     peak_cluster_center[-1] == peak_clusters[-1][-1]))):
    if len(peak_clusters) > 0 and len(peak_clusters[-1]) < cluster_size:
        peak_clusters.pop()
        peak_cluster_center.pop()
    peak_cluster_center = np.array(peak_cluster_center, dtype=np.int32) + offset
    return peak_clusters, peak_cluster_center

def peak_clustering_preamble_detection_multi_process(corr, extf, cluster_size=3, acc_batch=10, thres_percent=0.98):
    if len(corr) % acc_batch != 0:
        pad_len = len(corr) - (len(corr) // acc_batch) * acc_batch
        pad = torch.zeros(pad_len, dtype=torch.float32)
        corr = torch.concatenate([corr, pad])
    corr_batch_len = int(len(corr) // acc_batch)
    corr = torch.cat([corr, torch.zeros(16*extf*config.sample_pre_symbol, dtype=torch.float32)])
    peak_corr = corr.unfold(0, corr_batch_len+16*extf*config.sample_pre_symbol, corr_batch_len).numpy()
    res = []
    for i in range(acc_batch):
        res.append(process_pool.submit(peak_clustering_preamble_detection_1d, peak_corr[i, :], extf, cluster_size, i*corr_batch_len, thres_percent))
    return res

def multi_process_res_process(res):
    pc_centers = []
    for i in range(len(res)):
        _, pc = res[i].result()
        if len(pc_centers) > 0:
            if len(pc) > 0:
                new_min_val = pc[0]
                overlap_idx = np.where(pc_centers[-1] >= new_min_val)[0]
                if len(overlap_idx) > 0:
                    pc_centers[-1] = pc_centers[-1][:overlap_idx[0]]
                pc_centers.append(pc)
        else:
            pc_centers.append(pc)
    pc_centers = np.concatenate(pc_centers, dtype=np.int32)
    return pc_centers

# For phase difference, the correspongding sample point in the 
def peak_clustering_preamble_detection(corrlation, extf, cluster_size=3, promin_thres=80):
    max_diff = extf * config.sample_pre_symbol * 8
    peak_cluster_list = []
    peak_cluster_center_list = []
    for i in range(corrlation.shape[0]):
        peaks, properties = scipy.signal.find_peaks(corrlation[i, :], prominence=0, height=0)
        # A box-chart like 
        promin = properties["prominences"]
        promin_thres = np.percentile(promin, promin_thres)
        # promin_thres = promin_precentile
        height = properties["peak_heights"]
        height_thres = np.percentile(height, promin_thres)
        peaks = peaks[np.where((promin>promin_thres)&(height>height_thres))]
        # print(promin_thres, height_thres)

        if len(peaks) > 0:
            peak_clusters = [[peaks[0]]]
            peak_cluster_center = [peaks[0]] # should be the peak that with maximum value
            for j in range(1, len(peaks)):
                # The peak can be clustered into the last cluster
                if peaks[j] - peak_cluster_center[-1] < max_diff:
                    # This peak is higher than the original center
                    # We consider it as the new center
                    if corrlation[i, peaks[j]] > corrlation[i, peak_cluster_center[-1]]:
                        peak_cluster_center[-1] = peaks[j]
                    # Otherwise, we juse append the peak into the cluster
                    peak_clusters[-1].append(peaks[j])
                # This peak is far way from the former cluster
                # we juse take it as a new cluster 
                else:
                    # If last cluster is too small, we can consider it as a noise
                    # Another exceptation is that the center is on one-side
                    # (cluster_size > 1 and (peak_cluster_center[-1] == peak_clusters[-1][0] or \
                    #     peak_cluster_center[-1] == peak_clusters[-1][-1]))
                    if len(peak_clusters[-1]) < cluster_size:
                        peak_clusters.pop()
                        peak_cluster_center.pop()
                    peak_clusters.append([peaks[j]])
                    peak_cluster_center.append(peaks[j])
        else:
            peak_clusters = []
            peak_cluster_center = [-1]
        
        # Check wether the last cluster is noise or not
        # (len(peak_clusters[-1]) < cluster_size or \
        #     (cluster_size > 1 and (peak_cluster_center[-1] == peak_clusters[-1][0] or \
        #     peak_cluster_center[-1] == peak_clusters[-1][-1]))):
        if len(peak_clusters) > 0 and len(peak_clusters[-1]) < cluster_size:
            peak_clusters.pop()
            peak_cluster_center.pop()
        peak_cluster_list.append(peak_clusters)
        peak_cluster_center_list.append(peak_cluster_center)
    return peak_cluster_list, peak_cluster_center_list

    
def plot_peaks(corrlation, phase_cum, peak_cluster_list, peak_cluster_center_list, extf, prefix=""):
    plt_num = 20
    # if prefix.find("phase_diff") != -1:
    #     plt_num = 50
    for i in range(plt_num):
        peak_cluster = peak_cluster_list[i]
        peak_cluster_centers = peak_cluster_center_list[i]
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(corrlation[i, :])
        for j in range(len(peak_cluster)):
            plt.plot(np.array(peak_cluster[j]), corrlation[i, np.array(peak_cluster[j])], marker='.', markersize=10)
            plt.plot(peak_cluster_centers[j], corrlation[i, peak_cluster_centers[j]], marker='.', markersize=20, color="red")
        plt.xlim(0, phase_cum.shape[1])
        plt.title("Corrlations")
        plt.subplot(2, 1, 2)
        
        for j in range(len(peak_cluster)):
            idx = np.array([peak_cluster_centers[j]+k*config.sample_pre_symbol*extf for k in range(9)], dtype=np.int64)
            plt.plot(idx, phase_cum[i, idx], marker='.', markersize=10)
        plt.plot(phase_cum[i, :])
        plt.xlim(0, phase_cum.shape[1])
        plt.title("Phase cum")
        plt.savefig("./figs/preamble_detection/" + prefix + "peaks_" + str(i) + ".pdf")
        plt.close()

def preamble_add_noise(raw_signal, noise_signal, extf, peak_centers, snr):
    signal_len = raw_signal.shape[1]
    if noise_signal.shape[0] < raw_signal.shape[0]:
        noise_signal_tmp = np.zeros((raw_signal.shape[0], raw_signal.shape[1]), dtype=np.complex64)
        for i in range(raw_signal.shape[0]):
            noise_signal_tmp[i, :] = noise_signal[i % noise_signal.shape[0], (i // noise_signal.shape[0]) * raw_signal.shape[1]: (i // noise_signal.shape[0] + 1) * raw_signal.shape[1]]
        noise_signal = noise_signal_tmp
    else:
        noise_signal = noise_signal[: raw_signal.shape[0], : raw_signal.shape[1]]

    # Then we calculate the power of the preamble signal
    segment_len = 8 * config.sample_pre_symbol * extf
    real_signal_power = np.zeros(raw_signal.shape[0], dtype=np.float32)
    imag_signal_power = np.zeros(raw_signal.shape[0], dtype=np.float32)
    total_signal_len = np.zeros(raw_signal.shape[0], dtype=np.float32)
    for i in range(raw_signal.shape[0]):
        for center in peak_centers[i]:
            real_signal_power[i] += np.sum(np.real(raw_signal[i, center:center+segment_len])**2)
            imag_signal_power[i] += np.sum(np.imag(raw_signal[i, center:center+segment_len])**2)
        total_signal_len[i] = segment_len * len(peak_centers[i])
    real_signal_power /= total_signal_len
    imag_signal_power /= total_signal_len
    real_signal_power = real_signal_power.reshape(-1, 1)
    imag_signal_power = imag_signal_power.reshape(-1, 1)
    # Get the power of noise
    real_noise = np.real(noise_signal)
    imag_noise = np.imag(noise_signal)
    real_noise_power = (np.sum(real_noise ** 2, 1)).reshape(-1, 1) / signal_len
    imag_noise_power = (np.sum(imag_noise ** 2, 1)).reshape(-1, 1) / signal_len
    real_signal_variance = (np.power(10., (snr/10))) * real_noise_power
    imag_signal_variance = (np.power(10., (snr/10))) * imag_noise_power
    real_signal = np.sqrt(real_signal_variance / real_signal_power) * np.real(raw_signal) + real_noise
    imag_signal = np.sqrt(imag_signal_variance / imag_signal_power) * np.imag(raw_signal) + imag_noise
    signal_with_noise = np.zeros((raw_signal.shape[0], raw_signal.shape[1]), dtype=np.complex64)
    signal_with_noise.real = real_signal
    signal_with_noise.imag = imag_signal
    # return signal_with_noise
    b, a = scipy.signal.butter(8, 2e6 / config.sample_rate, "lowpass")
    filtered_signal = scipy.signal.filtfilt(b, a, signal_with_noise, 1)
    return filtered_signal

def preamble_detection_stft(signal_ori, extf, window_size_sym=1):
    # We copy the memory since the following operations will change the data
    signal = torch.tensor(signal_ori)
    # signal = torch.tensor(signal_seg)
    window_size = window_size_sym * config.sample_pre_symbol
    # Calcualte the STFT of the signal
    # If the extension is smaller than the window size, we pad 0s after each segment
    if extf >= window_size_sym:
        signal_window = signal.unfold(1, window_size, 1)
    else:
        signal_window = signal.unfold(1, extf * config.sample_pre_symbol, 1)
        left_pads_num = (window_size_sym * config.sample_pre_symbol - extf * config.sample_pre_symbol) // 2
        right_pads_num = window_size_sym * config.sample_pre_symbol - extf * config.sample_pre_symbol - left_pads_num
        left_pad = torch.zeros((signal_window.shape[0], signal_window.shape[1], left_pads_num), dtype=torch.complex64)
        right_pad = torch.zeros((signal_window.shape[0], signal_window.shape[1], right_pads_num), dtype=torch.complex64)
        signal_window = torch.cat((left_pad, signal_window, right_pad), dim=2)
    # Remove the DC value
    signal_mean = torch.mean(signal_window, dim=2).unsqueeze(2)
    signal_window = signal_window - signal_mean
    hanning_windows = torch.hann_window(window_size)
    signal_window = signal_window * hanning_windows
    # Start STFT, this tensor should be [batch, freq, time]
    stft_res = torch.fft.fft(signal_window, dim=2).transpose(1, 2).contiguous()
    # Combine the high and low frequency
    comb_res = torch.zeros(stft_res.shape[0], 2, stft_res.shape[2], dtype=torch.float32)
    freq_len = stft_res.shape[1] // 2
    if stft_res.shape[1] % 2 == 0:
        comb_res[:, 0, :] = torch.sum(torch.abs(stft_res[:, :freq_len, :]), dim=1)
        comb_res[:, 1, :] = torch.sum(torch.abs(stft_res[:, freq_len:, :]), dim=1)
    else:
        comb_res[:, 0, :] = torch.sum(torch.abs(stft_res[:, :freq_len, :]), dim=1)
        comb_res[:, 1, :] = torch.sum(torch.abs(stft_res[:, (freq_len+1):, :]), dim=1)
    comb_res = comb_res.unsqueeze(1)
    # Then we start to calculate the corrlaton between the current time-frequency plot and the target
    stft_mask = torch.zeros(2, config.sample_pre_symbol * extf * 8)
    pa_bits = np.concatenate([preamble_bits, target_aa_bit])
    for i in range(8):
        if pa_bits[i] % 2 == 0:
            stft_mask[0, i*extf*config.sample_pre_symbol: (i+1)*extf*config.sample_pre_symbol] = -1
            stft_mask[1, i*extf*config.sample_pre_symbol: (i+1)*extf*config.sample_pre_symbol] = 1
        else:
            stft_mask[0, i*extf*config.sample_pre_symbol: (i+1)*extf*config.sample_pre_symbol] = 1
            stft_mask[1, i*extf*config.sample_pre_symbol: (i+1)*extf*config.sample_pre_symbol] = -1
    stft_mask = stft_mask.unsqueeze(0).unsqueeze(0)
    corrlation = F.conv2d(comb_res, stft_mask).squeeze(1).squeeze(1).numpy()
    
    pc, pc_center = peak_clustering_preamble_detection(corrlation, extf, cluster_size=3)
    # if plt_flag:
    #     phase_cum = np.cumsum(cal_phase_diff(signal_ori), axis=1)
    #     plot_peaks(corrlation, phase_cum, pc, pc_center, extf, prefix="stft_" + str(window_size_sym) + "_")
    # Plot the STFT fig
    # comb_res = comb_res.squeeze(1).numpy()
    # for i in range(10):
    #     plt.figure(figsize=(100,8))
    #     plt.subplot(2, 1, 1)
    #     plt.imshow(np.abs(comb_res[i, :, :]), cmap="hot", aspect='auto')
    #     plt.subplot(2, 1, 2)
    #     plt.plot(phase_cum[i, :])
    #     plt.xlim([0, comb_res.shape[2]])
    #     plt.savefig("./figs/preamble_detection/stft_plt_" + str(i) + ".pdf")
    #     plt.close()
    return pc_center, corrlation

def preamble_detection_phase_diff(signal_ori, extf, mask=None, cluster_size=5):
    if mask is None:
        mask = get_preamble_mask(extf)
    phase_diff = cal_phase_diff(signal_ori)
    corrlation = cal_corrlation(phase_diff, mask)
    pc, pc_center = peak_clustering_preamble_detection(corrlation, extf, cluster_size)
    if plt_flag:
        phase_cum = np.cumsum(phase_diff, axis=1)
        plot_peaks(corrlation, phase_cum, pc, pc_center, extf, prefix="phase_diff_")
    # Each phase diff at index i is calculated by the i and i+1 sample point
    # if phase_diff>0, the i+1 sample point is 1, and if phase_diff<0, the i+1 sample point is 0
    # sample pint: 0   1   2   3   4   5
    # phase diff:    0   1   2   3   4
    # for i in range(len(pc_center)):
    #     for j in range(len(pc_center[i])):
    #         pc_center[i][j] += 1
    return pc_center, corrlation

def preamble_detection(signal_ori, extf, cluster_size=3, pa_thres=4, full=False, thres_percent=98):
    preamble_mask = get_preamble_mask(extf)
    preamble_aa_mask = get_preamble_aa_mask(extf, full)

    phase_diff = cal_phase_diff(signal_ori)
    phase_diff = np.hstack([phase_diff, np.zeros([phase_diff.shape[0], 1], dtype=np.float32)])
    bound = extf * 4 * config.sample_pre_symbol
    filtered_pcs = []
    corrs = []
    pa_corrs = []
    for i in range(signal_ori.shape[0]):
        phase_diff_tensor = torch.from_numpy(phase_diff[i:i+1, :])
        ss = time.time()
        corr = cal_corrlation(phase_diff_tensor, preamble_mask)[0]
        ee = time.time()
        print("preamble conv ", ee-ss)
        # preamble_aa_res = process_pool.submit()
        pc_centers_res = peak_clustering_preamble_detection_multi_process(corr, extf, cluster_size, acc_batch=11, thres_percent=thres_percent)
        pc_centers = multi_process_res_process(pc_centers_res)
        print("peak detection", time.time()-ee)
        pa_phase_diff_segs = []
        avi_idx = [] 
        ss = time.time()
        for j in range(len(pc_centers)):
            pc = pc_centers[j]
            st = pc-bound
            et = pc + bound + preamble_aa_mask.shape[2]
            if st >=0 and et < phase_diff_tensor.shape[1]:
                pa_phase_diff_segs.append(phase_diff_tensor[0, st:et])
                avi_idx.append(j)
        pa_phase_diff_segs = torch.vstack(pa_phase_diff_segs)
        pc_centers = pc_centers[avi_idx]
        pa_corr = cal_corrlation(pa_phase_diff_segs, preamble_aa_mask).numpy()
        print("pa conv", time.time()-ss)

        # Theoritically, the corrlation of Preamble+AA is 5x higher than the Preamble 
        # corr = corr[pc_centers].numpy()
        scale = (np.max(pa_corr, axis=1)[0] / corr[pc_centers].numpy())
        scale_idx = np.where((scale>=pa_thres) & (scale<10))[0]
        pc = pc_centers[scale_idx]
        avi_pa_corr = pa_corr[scale_idx]

        pa_corr_max = np.max(avi_pa_corr, axis=1)
        pa_corr_mean = np.mean(avi_pa_corr, axis=1)
        amp_scale = pa_corr_max / pa_corr_mean
        valid_idx = np.where(amp_scale > 1.5)[0]
        avi_pa_corr = avi_pa_corr[valid_idx]
        pc = pc[valid_idx]

        pa_max_idx = np.argmax(avi_pa_corr, axis=1)
        pa_real_idx = pa_max_idx + pc - bound
        pc_diff = np.abs(pa_real_idx - pc)
        filtered_idx = np.where(pc_diff < (config.sample_pre_symbol * extf / 2))[0]
        filtered_pcs.append(pa_real_idx[filtered_idx])
        corrs.append(corr)
        pa_corrs.append(avi_pa_corr[filtered_idx])
    return filtered_pcs, corrs, pa_corrs


def preamble_detection_arcphase(signal_ori, extf, cluster_size=3, pa_thres=4, full=False, thres_percent=98):
    preamble_mask = get_preamble_mask(extf)
    preamble_aa_mask = get_preamble_aa_mask(extf, full)

    bound = extf * 8 * config.sample_pre_symbol
    filtered_pcs = []
    corrs = []
    pa_corrs = []
    phase_diff_tensor = torch.from_numpy(cal_phase_diff(signal_ori))
    for i in range(signal_ori.shape[0]):
        ss = time.time()
        corr = cal_corrlation(phase_diff_tensor, preamble_mask)[0]
        ee = time.time()
        print("preamble conv ", ee-ss)
        # preamble_aa_res = process_pool.submit()
        pc_centers_res = peak_clustering_preamble_detection_multi_process(corr, extf, cluster_size, acc_batch=11, thres_percent=thres_percent)
        pc_centers = multi_process_res_process(pc_centers_res)
        print("peak detection", time.time()-ee)
        pa_phase_diff_segs = []
        avi_idx = []
        ss = time.time()
        for j in range(len(pc_centers)):
            pc = pc_centers[j]
            st = pc - bound
            et = pc + bound + preamble_aa_mask.shape[2] + 1
            if st >=0 and et < signal_ori.shape[1]:
                pa_phase_diff_segs.append(signal_ori[i, st:et])
                avi_idx.append(j)
        pa_phase_diff_segs = cal_phase_diff_arc(np.vstack(pa_phase_diff_segs))
        pa_phase_diff_segs = torch.from_numpy(pa_phase_diff_segs)
        pc_centers = pc_centers[avi_idx]
        pa_corr = cal_corrlation(pa_phase_diff_segs, preamble_aa_mask).numpy()
        print("pa conv", time.time()-ss)

        scale = 64 / extf
        thres = (3000 / scale) * (1 - scale * 0.0012)

        max_pa = np.max(pa_corr, axis=1)
        scale_idx = np.where(max_pa >= thres)[0]
        pc = pc_centers[scale_idx]
        avi_pa_corr = pa_corr[scale_idx]

        # pa_corr_max = np.max(avi_pa_corr, axis=1)
        # pa_corr_mean = np.mean(avi_pa_corr, axis=1)
        # amp_scale = pa_corr_max / pa_corr_mean
        # valid_idx = np.where(amp_scale > 1.5)[0]
        # avi_pa_corr = avi_pa_corr[valid_idx]
        # pc = pc[valid_idx]

        pa_max_idx = np.argmax(avi_pa_corr, axis=1)
        pa_real_idx = pa_max_idx + pc - bound
        pc_diff = np.abs(pa_real_idx - pc)
        filtered_idx = np.where(pc_diff < (config.sample_pre_symbol * extf / 2))[0]
        filtered_pcs.append(pa_real_idx[filtered_idx])
        corrs.append(corr)
        pa_corrs.append(avi_pa_corr[filtered_idx])
    return filtered_pcs, corrs, pa_corrs

def pkt_slice(signal, preamble_idx, extf, data_len, extra=config.extra, full=False):
    pkts = []
    if extf > 1:
        bits_pre_pdu = 2040 // extf
    else:
        bits_pre_pdu = 265 * 8
    res_len = 2040 - bits_pre_pdu * extf
    symbol_len = extf * config.sample_pre_symbol

    for i in range(len(preamble_idx)):
        pkt_sym = []
        cp = []
        cur_ptr = preamble_idx[i]
        for j in range(data_len):
            if j > 0 and j % bits_pre_pdu == 0:
                if full:
                    cur_ptr += (150 + 80 + res_len) * config.sample_pre_symbol
                else:
                    cur_ptr += (150 + 80) * config.sample_pre_symbol
            et = cur_ptr + symbol_len
            pkt_sym.append(signal[cur_ptr-extra:et+extra])
            cp.append(cur_ptr)
            cur_ptr = et
        pkts.append(np.array(pkt_sym))
    return np.array(pkts)

def preamble_detection_aa_check(signal_ori, f_signal, extf, pa_thres=2, cluster_size=5, full=False, thres_percent=98, getdet=False, model_name=None):
    preamble_mask = get_preamble_mask(extf)
    preamble_aa_mask = get_preamble_aa_mask(extf, full)
    phase_diff = cal_phase_diff(f_signal)
    phase_diff = np.hstack([phase_diff, np.zeros([phase_diff.shape[0], 1], dtype=np.float32)])

    bound = extf * 4 * config.sample_pre_symbol
    filtered_pcs = []
    stft_filtered_pcs = []
    det_nums = []
    for i in range(signal_ori.shape[0]):
        phase_diff_tensor = torch.from_numpy(phase_diff[i:i+1, :])
        ss = time.time()
        corr = cal_corrlation(phase_diff_tensor, preamble_mask)[0]
        ee = time.time()
        print("preamble conv ", ee-ss)
        # preamble_aa_res = process_pool.submit()
        pc_centers_res = peak_clustering_preamble_detection_multi_process(corr, extf, cluster_size, acc_batch=11, thres_percent=thres_percent)
        pc_centers = multi_process_res_process(pc_centers_res)
        print("peak detection", time.time()-ee)
        
        pa_phase_diff_segs = []
        avi_idx = [] 
        ss = time.time()
        for j in range(len(pc_centers)):
            pc = pc_centers[j]
            st = pc-bound
            et = pc + bound + preamble_aa_mask.shape[2]
            if st >=0 and et < phase_diff_tensor.shape[1]:
                pa_phase_diff_segs.append(phase_diff_tensor[0, st:et])
                avi_idx.append(j)
        pa_phase_diff_segs = torch.vstack(pa_phase_diff_segs)
        pc_centers = pc_centers[avi_idx]
        pa_corr = cal_corrlation(pa_phase_diff_segs, preamble_aa_mask).numpy()
        print("pa conv", time.time()-ss)

        # Theoritically, the corrlation of Preamble+AA is 5x higher than the Preamble 
        # corr = corr[pc_centers].numpy()
        # scale = (np.max(pa_corr, axis=1)[0] / corr[pc_centers].numpy())
        # scale_idx = np.where((scale>=pa_thres) & (scale<10))[0]
        # pc = pc_centers[scale_idx]
        # avi_pa_corr = pa_corr[scale_idx]
        # pa_corr_max = np.max(avi_pa_corr, axis=1)
        # pa_corr_mean = np.mean(avi_pa_corr, axis=1)
        # amp_scale = pa_corr_max / pa_corr_mean
        # valid_idx = np.where(amp_scale > 1.5)[0]
        # avi_pa_corr = avi_pa_corr[valid_idx]
        # pc = pc[valid_idx]
        pc = pc_centers
        avi_pa_corr = pa_corr
        pa_max_idx = np.argmax(avi_pa_corr, axis=1)
        pa_real_idx = pa_max_idx + pc - bound
        pc_diff = np.abs(pa_real_idx - pc)
        filtered_idx = np.where(pc_diff < (config.sample_pre_symbol * extf / 2))[0]
        preamble_idx = pa_real_idx[filtered_idx]

        print("{} possible preamble".format(len(preamble_idx)))

        pkts = pkt_slice(signal_ori[i], preamble_idx, extf, 40, full=full)
        syms = pkts.reshape([-1, pkts.shape[2]])
        valid_idx, stft_valid_idx, det_num = dnn.dnn_decode_bits(syms, extf, pa_bits, getdet=True, model_name=model_name)
        if len(valid_idx) > 0:
            filtered_pcs.append(preamble_idx[valid_idx])
        else:
            filtered_pcs.append(np.array([], dtype=np.int32))
        if len(stft_valid_idx) > 0:
            stft_filtered_pcs.append(preamble_idx[stft_valid_idx])
        else:
            stft_filtered_pcs.append(np.array([], dtype=np.int32))
        det_nums.append(det_num)
    if getdet:
        return filtered_pcs, stft_filtered_pcs, det_nums
    else:
        return filtered_pcs

def preamble_detection_aa_check_test(signal_ori, f_signal, extf, pa_thres=2, cluster_size=5, full=False, thres_percent=98, model_name=None):
    preamble_mask = get_preamble_mask(extf)
    preamble_aa_mask = get_preamble_aa_mask(extf, full)
    phase_diff = cal_phase_diff(f_signal)
    phase_diff = np.hstack([phase_diff, np.zeros([phase_diff.shape[0], 1], dtype=np.float32)])

    bound = extf * 4 * config.sample_pre_symbol
    pkt_detect_rate = []
    for i in range(signal_ori.shape[0]):
        phase_diff_tensor = torch.from_numpy(phase_diff[i:i+1, :])
        ss = time.time()
        corr = cal_corrlation(phase_diff_tensor, preamble_mask)[0]
        ee = time.time()
        print("preamble conv ", ee-ss)
        # preamble_aa_res = process_pool.submit()
        pc_centers_res = peak_clustering_preamble_detection_multi_process(corr, extf, cluster_size, acc_batch=11, thres_percent=thres_percent)
        pc_centers = multi_process_res_process(pc_centers_res)
        print("peak detection", time.time()-ee)
        
        pa_phase_diff_segs = []
        avi_idx = [] 
        ss = time.time()
        for j in range(len(pc_centers)):
            pc = pc_centers[j]
            st = pc-bound
            et = pc + bound + preamble_aa_mask.shape[2]
            if st >=0 and et < phase_diff_tensor.shape[1]:
                pa_phase_diff_segs.append(phase_diff_tensor[0, st:et])
                avi_idx.append(j)
        pa_phase_diff_segs = torch.vstack(pa_phase_diff_segs)
        pc_centers = pc_centers[avi_idx]
        pa_corr = cal_corrlation(pa_phase_diff_segs, preamble_aa_mask).numpy()
        print("pa conv", time.time()-ss)

        pc = pc_centers
        avi_pa_corr = pa_corr
        pa_max_idx = np.argmax(avi_pa_corr, axis=1)
        pa_real_idx = pa_max_idx + pc - bound
        pc_diff = np.abs(pa_real_idx - pc)
        filtered_idx = np.where(pc_diff < (config.sample_pre_symbol * extf / 2))[0]
        preamble_idx = pa_real_idx[filtered_idx]

        print("{} possible preamble".format(len(preamble_idx)))

        pkts = pkt_slice(signal_ori[i], preamble_idx, extf, 40, full=full)
        syms = pkts.reshape([-1, pkts.shape[2]])
        rates = dnn.dnn_preamble_detection_test(syms, extf, pa_bits, model_name)
        pkt_detect_rate.append(rates)
    return pkt_detect_rate

def preamble_detection_aa_check_native(signal_ori, f_signal, extf, pa_thres=2, cluster_size=5, full=False, thres_percent=98):
    preamble_mask = get_preamble_mask(extf)
    preamble_aa_mask = get_preamble_aa_mask(extf, full)
    phase_diff = cal_phase_diff(f_signal)
    phase_diff = np.hstack([phase_diff, np.zeros([phase_diff.shape[0], 1], dtype=np.float32)])

    bound = extf * 4 * config.sample_pre_symbol
    filtered_pcs = []
    for i in range(signal_ori.shape[0]):
        phase_diff_tensor = torch.from_numpy(phase_diff[i:i+1, :])
        ss = time.time()
        corr = cal_corrlation(phase_diff_tensor, preamble_mask)[0]
        ee = time.time()
        print("preamble conv ", ee-ss)
        # preamble_aa_res = process_pool.submit()
        pc_centers_res = peak_clustering_preamble_detection_multi_process(corr, extf, cluster_size, acc_batch=11, thres_percent=thres_percent)
        pc_centers = multi_process_res_process(pc_centers_res)
        print("peak detection", time.time()-ee)
        
        pa_phase_diff_segs = []
        avi_idx = [] 
        ss = time.time()
        for j in range(len(pc_centers)):
            pc = pc_centers[j]
            st = pc-bound
            et = pc + bound + preamble_aa_mask.shape[2]
            if st >=0 and et < phase_diff_tensor.shape[1]:
                pa_phase_diff_segs.append(phase_diff_tensor[0, st:et])
                avi_idx.append(j)
        pa_phase_diff_segs = torch.vstack(pa_phase_diff_segs)
        pc_centers = pc_centers[avi_idx]
        pa_corr = cal_corrlation(pa_phase_diff_segs, preamble_aa_mask).numpy()
        print("pa conv", time.time()-ss)

        pc = pc_centers
        avi_pa_corr = pa_corr
        pa_max_idx = np.argmax(avi_pa_corr, axis=1)
        pa_real_idx = pa_max_idx + pc - bound
        pc_diff = np.abs(pa_real_idx - pc)
        filtered_idx = np.where(pc_diff < (config.sample_pre_symbol * extf / 2))[0]
        preamble_idx = pa_real_idx[filtered_idx]

        print("{} possible preamble".format(len(preamble_idx)))

        pkts = pkt_slice(signal_ori[i], preamble_idx, extf, 40, full=full)
        syms = pkts.reshape([-1, pkts.shape[2]])
        _, valid_idx, _ = dnn.native_decode_bits(syms, extf, pa_bits)
        print(valid_idx)
        if len(valid_idx) > 0:
            filtered_pcs.append(preamble_idx[valid_idx])
        else:
            filtered_pcs.append(np.array([], dtype=np.int32))
    return filtered_pcs

def matched_filter(signal, extf, freq_dev=250e3):
    ref_signal, _ = utils.get_ref_signal(extf, freq_dev)
    ref_signal = np.flip(ref_signal).conjugate().copy()
    ref_signal = torch.tensor(ref_signal, device=device, dtype=torch.complex64).view(1, 1, -1).contiguous()  # [batch=1, chan=1, ref_signal_len]
    signal = torch.tensor(signal, device=device, dtype=torch.complex64).unsqueeze(1).contiguous()  # [batch, 1, signa_len]
    out = F.conv1d(signal, ref_signal).squeeze(1).cpu().numpy()
    return out 

def ground_turth(signal, extf, full, time_int=15):
    scale = 64 / extf
    thres = 3000 / scale
    print("thres=", thres)
    phase = utils.get_phase(signal)
    phase_diff = phase[:, 1:] - phase[:, 0:-1]
    preamble_aa_mask = get_preamble_aa_mask(extf, full)
    corr = cal_corrlation(phase_diff, preamble_aa_mask)
    if extf <= 2:
        phase_diff_amp = utils.get_phase_diff(signal)
        preamble_mask = get_preamble_mask(extf)
        corr_amp = cal_corrlation(phase_diff_amp, preamble_mask)
    p_idxs = []
    for i in range(signal.shape[0]):
        # print("thres=", thres)
        if extf <= 2:
            amp_thres = np.percentile(np.abs(corr_amp[i, :]), 90)
            c_p_idx = np.where((corr[i, :] >= thres) & (corr_amp[i, :] >= amp_thres))[0]
        else:
            c_p_idx = np.where(corr[i, :] >= thres)[0]

        p_diff = c_p_idx[1:] - c_p_idx[0:-1]
        seg_pos = np.where(p_diff > (time_int * 1000 * config.sample_pre_symbol))[0]
        last_pos = 0
        p_idx = []
        for j in range(len(seg_pos)):
            cur_pos = seg_pos[j] + 1
            p_idx_seg = c_p_idx[last_pos: cur_pos]
            corr_seg = corr[i, p_idx_seg]
            p_idx.append(p_idx_seg[np.argmax(corr_seg)])
            last_pos = cur_pos
        p_idxs.append(np.array(p_idx))
    # plt.figure(figsize=(12, 6))
    # plt.plot(corr[0, :])
    # plt.savefig("./figs/pd.pdf")
    return p_idxs, corr

plt_flag = False
# To warm-up
empty = torch.ones((1, 1, 100), device=device)
F.conv1d(empty, empty)

if __name__ == "__main__":
    extf = 16
    raw_data = np.load("./raw_data/0m_4c_16e_100n_0.npz")["arr_0"]
    raw_data = raw_data.flatten()[np.newaxis,:]  # int(3 * 64e4):int(6 * 64e4)
    filtered_signal = utils.filter(raw_data)
    p_idx, corr = ground_turth(filtered_signal, extf, True)
    p_idx = p_idx[0]
    print(p_idx)
    t_diff = (p_idx[1:] - p_idx[0:-1]) / 8e6
    plt.figure(figsize=(12, 12))
    plt.subplot(211)
    plt.plot(t_diff)
    plt.subplot(212)
    plt.plot(corr[0])
    plt.savefig("./figs/time.pdf")

    # p_idx, corr, pa_corr = preamble_detection(filtered_signal, 64, cluster_size=3, full=True)
    # print(p_idx[0])
    # phase_cum = utils.get_phase_cum(filtered_signal)
    # print(phase_cum.shape)
    # preamble_aa_mask = get_preamble_aa_mask(64, True)
    # p_corr = cal_corrlation(utils.get_phase_diff(filtered_signal), preamble_aa_mask)

    # plt.figure(figsize=(12, 12))
    # plt.subplot(211)
    # plt.plot(phase_cum[0, :])
    # for i in range(len(p_idx[0])):
    #     plt.plot([p_idx[0][i] for _ in range(2)], [0, -3.5])
    # plt.subplot(212) 
    # plt.plot(p_corr[0,:])
    # plt.savefig("./figs/p.pdf")



