import numpy as np
import config
from scramble_table import *
import cv2
import scipy
import torch

# buffer = ringbuffer(config.num_samps, np.complex64)
power_of_2 = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
preamble = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
b, a = scipy.signal.butter(8, 1.2e6 / config.sample_rate, "lowpass")
b1m, a1m = scipy.signal.butter(8, 2e6 / config.sample_rate, "lowpass")
down_sample_b, down_sample_a = scipy.signal.butter(8, 1e6 / config.down_sample_rate, "lowpass")

def filter(raw_signal):
    pad_len = max(len(b), len(a))
    if raw_signal.ndim == 2:
        if raw_signal.shape[1] <= 3 * pad_len:
            pad_len = (raw_signal.shape[1] // pad_len) * pad_len
        else:
            pad_len = 3 * pad_len
        return scipy.signal.filtfilt(b, a, raw_signal, 1, padlen=pad_len)  
    else:
        if len(raw_signal) <= 3 * pad_len:
            pad_len = (len(raw_signal) // pad_len) * pad_len
        else:
            pad_len = 3 * pad_len
        return scipy.signal.filtfilt(b, a, raw_signal, padlen=pad_len)
    
def down_sample_filter(raw_signal):
    if raw_signal.ndim == 2:
        return scipy.signal.filtfilt(down_sample_b, down_sample_a, raw_signal, 1)
    else:
        return scipy.signal.filtfilt(down_sample_b, down_sample_a, raw_signal)

def get_freq_with_chan_num(chan):
    if chan == 37:
        return 2402000000
    elif chan == 38:
        return 2426000000
    elif chan == 39:
        return 2480000000
    elif chan >= 0 and chan <= 10:
        return 2404000000 + chan * 2000000
    elif chan >= 11 and chan <= 36:
        return 2428000000 + (chan - 11) * 2000000
    else:
        return 0xffffffffffffffff

def get_aa_bits(aa_byte):
    aa_bits = np.zeros(32, dtype=np.uint8)
    for i in range(32):
        if (aa_byte >> i) & 1:
            aa_bits[i] = 1
    return aa_bits

def search_sequence(bits, target_bits):
    res = cv2.matchTemplate(bits, target_bits, cv2.TM_SQDIFF)
    idx = np.where(res==0)[0]
    if len(idx) > 0:
        return idx
    else:
        return None

def raw_signal_segment(data, timestamp, target_aa_bits):
    i0 = np.real(data[:-1])
    q0 = np.imag(data[:-1])
    i1 = np.real(data[1:])
    q1 = np.imag(data[1:])
    phase_diff = i0 * q1 - i1 * q0
    phase_len = len(phase_diff)
    bits = np.zeros(int(np.ceil(phase_len / config.sample_pre_symbol)), dtype=np.uint8)
    for i in range(config.sample_pre_symbol):
        bits[:] = 0
        sample_len = (phase_len - i) // config.sample_pre_symbol
        p = phase_diff[i: i+sample_len*config.sample_pre_symbol].reshape(-1, config.sample_pre_symbol)
        vote = np.sum(p, 1)
        # vote = phase_diff[i: i+sample_len*config.sample_pre_symbol: config.sample_pre_symbol]
        bits[np.where(vote>0)[0]] = 1
        preamble_idx = search_sequence(bits, target_aa_bits)
        # print("preamble_idx ", preamble_idx)
        if preamble_idx is not None:
            for j in range(len(preamble_idx)):
                idx = (preamble_idx[j] - 18) * config.sample_pre_symbol
                end_idx = idx + config.sample_pre_symbol * 200 * 8 + config.sample_pre_symbol
                empty_idx = (preamble_idx[j] - 18) * config.sample_pre_symbol + config.sample_pre_symbol * 200 * 8 + config.sample_pre_symbol
                empty_end_idx = empty_idx + config.sample_pre_symbol * 200 * 8 + config.sample_pre_symbol
                if idx >=0 and empty_end_idx < len(data):
                    # bits_data = bits[preamble_idx[j]: preamble_idx[j] + 166 * 8].reshape(-1, 8)
                    # bytes_data = np.sum(bits_data * power_of_2, 1, dtype=np.uint8)
                    # bytes_data[4: 6] ^= scramble_table[8][:2]
                    # bytes_data = bytearray(bytes_data)
                    # print(bytes_data.hex())
                    raw_data = data[idx: end_idx]
                    empty_raw_data = data[empty_idx: empty_end_idx]
                    # print(idx, end_idx)
                    return raw_data, empty_raw_data, timestamp[0] + idx * (1 / config.sample_rate)
                else:
                    print("sample too short", idx, end_idx)
    print("not find")
    return None, None, -1

def signal_lpf(signals):
    pad_len = max(len(b), len(a))
    if signals.shape[1] <= 3 * pad_len:
        pad_len = (signals.shape[1] // pad_len) * pad_len
    else:
        pad_len = 3 * pad_len
    filtered_signal = scipy.signal.filtfilt(b, a, signals, axis=1, padlen=pad_len).ascontiguousarray()
    return filtered_signal


def awgn(raw_signal, snr, noise_signal=None, extra=0):
    if raw_signal.ndim == 1:
        raw_signal = raw_signal[np.newaxis, :]

    if noise_signal is None:
        noise_signal = np.zeros([raw_signal.shape[0], raw_signal.shape[1]+extra], dtype=np.complex64)
        noise_signal.real = np.random.randn(raw_signal.shape[0], raw_signal.shape[1]+extra)
        noise_signal.imag = np.random.randn(raw_signal.shape[0], raw_signal.shape[1]+extra)
    elif noise_signal.ndim == 1:
        noise_signal = noise_signal[np.newaxis, :]

    if noise_signal.shape[0] < raw_signal.shape[0]:
        noise_signal_tmp = np.zeros((raw_signal.shape[0], raw_signal.shape[1]+extra), dtype=np.complex64)
        for i in range(raw_signal.shape[0]):
            noise_signal_tmp[i, :] = noise_signal[i % noise_signal.shape[0], (i // noise_signal.shape[0]) * (raw_signal.shape[1] + extra): (i // noise_signal.shape[0] + 1) * (raw_signal.shape[1] + extra)]
        noise_signal = noise_signal_tmp
    else:
        noise_signal = noise_signal[: raw_signal.shape[0], : raw_signal.shape[1]+extra]
    noise_power = noise_signal.real ** 2 + noise_signal.imag ** 2
    noise_power = np.average(noise_power, axis=1)
    signal_power = raw_signal.real ** 2 + raw_signal.imag ** 2
    signal_power = np.average(signal_power, axis=1)
    variance = np.power(10, (snr/10)) * noise_power
    scaled_signal = np.sqrt(variance / signal_power)[:, np.newaxis] * raw_signal
    signal = noise_signal.copy()
    signal[:, extra:] += scaled_signal
    # amp = np.abs(signal)
    # signal_percent = np.percentile(amp, [percent, 100-percent], axis=1).T
    # intval = (signal_percent[:, 1] - signal_percent[:, 0]) / 2
    # signal = signal / intval[:, np.newaxis]
    return signal

def preamble_awgn(raw_signal, snr, centers, extf, noise_signal, extra=0):
    if raw_signal.ndim == 1:
        raw_signal = raw_signal[np.newaxis, :]

    noise_signal = noise_signal[: raw_signal.shape[0], : raw_signal.shape[1]+extra]
    noise_power = noise_signal.real ** 2 + noise_signal.imag ** 2
    noise_power = np.average(noise_power, axis=1)
    signal_power = np.zeros(raw_signal.shape[0])
    segment_len = 8 * config.sample_pre_symbol * extf
    for i in range(signal_power.shape[0]):
        seg = raw_signal[i, centers[i, 0]:centers[i, 0]+segment_len]
        seg_power = seg.real ** 2 + seg.imag ** 2
        signal_power[i] = np.average(seg_power)
    # signal_power = raw_signal.real ** 2 + raw_signal.imag ** 2
    # signal_power = np.average(signal_power, axis=1)
    variance = np.power(10, (snr/10)) * noise_power
    scaled_signal = np.sqrt(variance / signal_power)[:, np.newaxis] * raw_signal
    signal = noise_signal
    signal[:, extra:] += scaled_signal
    b1m, a1m = scipy.signal.butter(8, 2e6 / config.sample_rate, "lowpass")
    filtered_signal = scipy.signal.filtfilt(b1m, a1m, signal, 1)
    # amp = np.abs(signal)
    # signal_percent = np.percentile(amp, [percent, 100-percent], axis=1).T
    # intval = (signal_percent[:, 1] - signal_percent[:, 0]) / 2
    # signal = signal / intval[:, np.newaxis]
    return filtered_signal

# Add the WiFi interference
def wifi_awgn(raw_signal, snr, extf, p_idx, noise_signal):
    noise_power = noise_signal.real ** 2 + noise_signal.imag ** 2
    noise_power = np.average(noise_power)
    signal = []
    for p in p_idx:
        signal.append(raw_signal[p:p+extf*8*config.sample_pre_symbol])
    signal = np.concatenate(signal)
    signal_power = signal.real ** 2 + signal.imag ** 2
    signal_power = np.average(signal_power)
    variance = np.power(10, (snr/10)) * noise_power
    scaled_signal = np.sqrt(variance / signal_power) * raw_signal
    signal = noise_signal + scaled_signal
    return signal

def signal_normalize(raw_signal, percent=1):
    amp = np.abs(raw_signal)
    signal_percent = np.percentile(amp, [percent, 100-percent], axis=1).T
    intval = (signal_percent[:, 1] - signal_percent[:, 0]) / 2
    signal = raw_signal / intval[:, np.newaxis]
    return signal

def get_phase_diff(raw_signal):
    return raw_signal[:, 0:-1].real * raw_signal[:, 1:].imag - raw_signal[:, 0:-1].imag * raw_signal[:, 1:].real

def get_phase_cum(raw_signal):
    phase_diff = raw_signal[:, 0:-1].real * raw_signal[:, 1:].imag - raw_signal[:, 0:-1].imag * raw_signal[:, 1:].real
    phase_cum = np.cumsum(phase_diff, axis=1)
    return phase_cum

def get_phase_cum_1d(raw_signal):
    phase_diff = raw_signal[0:-1].real * raw_signal[1:].imag - raw_signal[0:-1].imag * raw_signal[1:].real
    phase_cum = np.cumsum(phase_diff)
    return phase_cum

def get_phase(raw_signal):
    angle = np.arctan2(raw_signal.imag, raw_signal.real)
    phase = np.unwrap(angle, axis=1)
    return phase

p2 = torch.tensor([[1, 2, 4, 8, 16, 32, 64, 128]], dtype=torch.uint8)
def bits_to_bytes(bits):
    bits = bits.reshape([-1, 8])
    bytes = torch.sum(bits * p2, dim=1)
    return bytes.numpy()

# It seems that there can be a very high peak at the begin and end of the raw signal.
# These peaks is caused by the LPF, since it does not handle the boundaries well
# We just ignore few sample points at very beginning and ending
def normalize(origin_data, zero_center=True, omitted_len=20):
    data = origin_data[:, omitted_len:origin_data.shape[1]-omitted_len]
    if np.iscomplexobj(origin_data):
        max_real = np.max(data.real, axis=1)
        min_real = np.min(data.real, axis=1)
        max_imag = np.max(data.imag, axis=1)
        min_imag = np.min(data.imag, axis=1)
        max_combo = np.hstack([max_real.reshape([-1, 1]), max_imag.reshape([-1, 1])])
        min_combo = np.hstack([min_real.reshape([-1, 1]), min_imag.reshape([-1, 1])])
        max_val = np.max(max_combo, axis=1)
        min_val = np.min(min_combo, axis=1)
    else:
        max_val = np.max(data, axis=1)
        min_val = np.min(data, axis=1)
    interval = (max_val - min_val).reshape([-1, 1])
    if zero_center:
        return origin_data / interval, interval.reshape([-1])
    else:
        return (origin_data - min_val) / interval, interval.reshape([-1])

def get_cfo(phase, preamble_idx, phase_amp=None):
    phase_slope = np.zeros(phase.shape[0], dtype=np.float64)
    phase_slope_amp = None
    if phase_amp is not None:
        phase_slope_amp = np.zeros(phase.shape[0], dtype=np.float64)
    for i in range(phase.shape[0]):
        if preamble_idx[0] == 0:
            continue
        upper_preamble_bit_idx = np.array([preamble_idx[i]+bit_idx*config.sample_pre_symbol for bit_idx in range(2, 9, 2)], dtype=np.int32)
        lower_preamble_bit_idx = np.array([preamble_idx[i]+bit_idx*config.sample_pre_symbol for bit_idx in range(1, 8, 2)], dtype=np.int32)
        upper_slope, _ = np.polyfit(upper_preamble_bit_idx, phase[i, upper_preamble_bit_idx], 1)
        lower_slope, _ = np.polyfit(lower_preamble_bit_idx, phase[i, lower_preamble_bit_idx], 1)
        phase_slope[i] = (upper_slope + lower_slope) / 2
        if phase_amp is not None:
            amp_upper_slope, _ = np.polyfit(upper_preamble_bit_idx, phase_amp[i, upper_preamble_bit_idx], 1)
            amp_lower_slope, _ = np.polyfit(lower_preamble_bit_idx, phase_amp[i, lower_preamble_bit_idx], 1)
            phase_slope_amp[i] = (amp_upper_slope + amp_lower_slope) / 2
        # if i < 10:
        #     plt.figure(figsize=(12, 6))
        #     plt.plot(phase[i, :])
        #     plt.savefig("./figs/cfo_2" + str(i) + ".pdf")
        #     plt.close()
        
    # for now, the cfo is the phase difference per sample point, translate it into frequency

    cfo = phase_slope * config.sample_rate / 2 / np.pi
    return cfo, phase_slope, phase_slope_amp
    
def cfo_compensate(signal, cfo):
    if signal.ndim == 1:
        signal = signal.reshape([1, -1])
        cfo = np.array([cfo])

    t = np.arange(0, signal.shape[1]) * (1 / config.sample_rate)
    time = np.array([t for _ in range(signal.shape[0])])
    cfo = cfo.reshape([-1, 1])
    freq_dev = 2 * np.pi * (-cfo) * time
    freq_dev_signal = np.exp(1j * freq_dev)
    return signal * freq_dev_signal

def cfo_compensate_1d(signal, cfo):
    t = np.arange(0, len(signal)) / config.sample_rate
    freq_dev = 2 * np.pi * (-cfo) * t
    return signal * np.exp(1j * freq_dev)

def frequency_deviation(signal, deviation):
    t = np.arange(0, signal.shape[1]) / config.sample_rate
    time = np.array([t for _ in range(signal.shape[0])])
    freq_dev = 2 * np.pi * deviation * time
    return signal * np.exp(1j * freq_dev)

def add_noise(raw_signal, noise_power):
    iq_noise = noise_power * np.random.normal(0, 1, (raw_signal.shape[0], raw_signal.shape[1], 2))
    noise_signal = np.zeros(raw_signal.shape, dtype=np.complex64)
    noise_signal[:, :].real = raw_signal[:, :].real + iq_noise[:, :, 0]
    noise_signal[:, :].imag = raw_signal[:, :].imag + iq_noise[:, :, 1]
    return noise_signal



# To gerneate a gaussian FIR for pulse shapping, same as gaussdesign in Matlab
def gaussdesign(bt=0.5, span=3, sps=20):
    filtLen = span * sps + 1
    t = np.linspace(-span/2, span/2, filtLen)
    alpha = np.sqrt(np.log(2) / 2) / bt
    h = (np.sqrt(np.pi)/alpha) * np.exp(-(t*np.pi/alpha)**2); 
    h = h / np.sum(h)
    return h

def gfsk_ref_signal(preamble=None, postfix=None, freq_dev=250e3, init_phase=0, sample_rate=config.sample_rate):
    sample_pre_symbol = int(sample_rate // 1e6)
    g_filter = gaussdesign(sps=sample_pre_symbol)
    if preamble is None:
        preamble = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    if postfix is None:
        postfix = np.array([0, 0, 0], dtype=np.int8)
    
    data = np.concatenate([preamble, postfix])
    # The GFSK first up-sample the data and turn it into NRZ code
    up_sampling = np.zeros((len(data)) * sample_pre_symbol, dtype=np.float64)
    for i in range(len(data)):
        up_sampling[i*sample_pre_symbol: (i+1)*sample_pre_symbol] = data[i] * 2 - 1
    
    # Use gaussian pulse shape filter 
    gamma_gfsk = scipy.signal.lfilter(g_filter, np.array([1], dtype=np.float64), up_sampling)
    gfsk_phase = np.zeros(len(gamma_gfsk))
    gfsk_phase[1:] = 2 * np.pi * freq_dev * (1 / sample_rate) * np.cumsum((gamma_gfsk[:-1] + gamma_gfsk[1:]) / 2)
    gfsk_phase += init_phase
    gfsk_phase = gfsk_phase[int(1.5*sample_pre_symbol):int((1.5+len(preamble))*sample_pre_symbol)]
    gfsk_phase = gfsk_phase[:] - gfsk_phase[0]
    t = np.arange(len(gfsk_phase)) / sample_rate
    return np.exp(1j * gfsk_phase), gfsk_phase, t

def get_ref_signal(extf, freq_dev=250e3, sample_rate=config.sample_rate):
    preamble = [0, 1, 0, 1, 0, 1, 0, 1]
    ext_preamble = []
    for i in range(len(preamble)):
        ext_preamble.append(np.array([preamble[i] for _ in range(extf)], dtype=np.int8))
    ext_preamble = np.concatenate(ext_preamble)
    ref_signal, _, t = gfsk_ref_signal(ext_preamble, freq_dev=freq_dev, sample_rate=sample_rate)
    return ref_signal, t

# The raw_signal_seg is combined with config.feat_len
def cal_SNR(raw_signal_seg, noise_seg, scale):
    preamble_raw = raw_signal_seg[:, config.feat_len:]
    phase = get_phase(preamble_raw)
    phase_diff_amp = get_phase_diff(preamble_raw)
    phase_diff_std = np.sin(phase[:, 1:] - phase[:, 0:-1])
    signal_power = np.average(phase_diff_amp / phase_diff_std, axis=1) / 2 * (scale ** 2)
    noise_power = np.zeros(raw_signal_seg.shape[0], dtype=np.float64)
    for i in range(raw_signal_seg.shape[0]):
        noise_power[i] = np.mean(np.abs(noise_seg[i]) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# def phase_diff_decode(raw_signal=None, phase_diff=None):
#     if raw_signal is not None and phase_diff is None:
#         phase_diff = raw_signal[:, :-1].real * raw_signal[:, 1:].imag - raw_signal[:, 1:].real * raw_signal[:, :-1].imag
#     bit_len = (phase_diff.shape[1] // config.sample_pre_symbol) * config.sample_pre_symbol
#     phase_diff = phase_diff[:, :bit_len]
#     phase_diff = phase_diff.reshape(phase_diff.shape[0], -1, config.sample_pre_symbol)
#     phase_sum = np.sum(phase_diff, 2)
#     bits = np.zeros((phase_diff.shape[0], phase_diff.shape[1]), dtype=np.uint8)
#     bits[np.where(phase_sum>0)] = 1
#     return bits

def stft_decode(stft_res):
    bits = torch.zeros(int(stft_res.shape[0]), dtype=torch.float32)
    amp = stft_res[:, 0, :, :] ** 2 + stft_res[:, 1, :, :] ** 2
    freq_sec_num = int(amp.shape[1]) // 2
    positive_amp = torch.sum(torch.sum(amp[:, :freq_sec_num, :], dim=2), dim=1)
    negative_amp = torch.sum(torch.sum(amp[:, freq_sec_num:, :], dim=2), dim=1)
    ones_idx = torch.where(positive_amp >= negative_amp)
    bits[ones_idx] = 1
    return bits

def phase_diff_decode(raw_chan):
    phase_diff = raw_chan[:, 2, :-1] * raw_chan[:, 3, 1:] - raw_chan[:, 2, 1:] * raw_chan[:, 3, :-1]
    phase_cum = torch.sum(phase_diff, dim=1)
    bits = torch.zeros(int(raw_chan.shape[0]), dtype=torch.float32)
    bits[torch.where(phase_cum>=0)] = 1
    return bits

def generate_white_noise(signal_shape):
    noise_signal = np.zeros(signal_shape, dtype=np.complex64)
    noise_signal.real = np.random.normal(0, 1, signal_shape)
    noise_signal.imag = np.random.normal(0, 1, signal_shape)
    return noise_signal

if __name__ == "__main__":
    data = np.load("./raw_data/single_preamble_0_chan=8_extf=" + str(128) + ".npz")["arr_0"]
    print(data.shape)
    s = list(data.shape)
    s[0] *= 8
    noise = generate_white_noise(s)
    np.save("./raw_data/white_noise_2.npy", noise)