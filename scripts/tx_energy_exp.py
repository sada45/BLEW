import sys
sys.path.append("./scripts")
import numpy as np
import opt.goodput_opt as go

detect_rate_file = "./output/packet_detection/awgn_packet_detection_{}_detail.csv"


extfs = [1, 2, 4, 8, 16, 32, 64]

def get_snr_ber_relation(train):
    blong = {}
    stft = {}
    pd = {}
    for extf in extfs:
        blong[extf] = {}
        stft[extf] = {}
        pd[extf] = {}
    if train:
        fp = "./output/white_noise_dnn_gain_train_2.csv"
    else:
        fp = "./output/white_noise_dnn_gain_2.csv"
    data = np.loadtxt(fp, delimiter=",")
    # 用原来的extf和snr要换一下
    for i in range(data.shape[0]):
        if int(data[i, 0]) <= 2:
            blong[int(data[i, 1])][int(data[i, 0])] = data[i, 3]
        else:
            blong[int(data[i, 1])][int(data[i, 0])] = data[i, 2]
        stft[int(data[i, 1])][int(data[i, 0])] = data[i, 3]
        pd[int(data[i, 1])][int(data[i, 0])] = data[i, 4]
    return blong, stft, pd

def get_detect_rate():
    train_dict = {}
    test_dict = {}
    stft_dict = {}
    for extf in extfs:
        train_dict[extf] = {}
        test_dict[extf] = {}
        stft_dict[extf] = {}
        fp = detect_rate_file.format(extf)
        data = np.loadtxt(fp, delimiter=",")
        snr = data[:, 1]
        train_dr = data[:, 2:5]
        test_dr = data[:, 5:8]
        for i in range(len(snr)):
            train_dict[extf][snr[i]] = train_dr[i, 2]
            test_dict[extf][snr[i]] = test_dr[i, 2]
            stft_dict[extf][snr[i]] = test_dr[i, 1]
    return train_dict, test_dict, stft_dict

train_dr, test_dr, stft_dr = get_detect_rate()
train_blong_bcr, _, _ = get_snr_ber_relation(True)
test_blong_bcr, test_stft_bcr, test_pd_bcr = get_snr_ber_relation(False)

def max_throughput(extf, payload_n_max):
    if extf == 1:
        mt = payload_n_max / (payload_n_max + 10 + 150 / 8)
    else:
        n_total = (payload_n_max + 10) * extf
        n_bit = np.floor(255 * 8 / extf)
        n_aip = np.ceil(n_total / n_bit)
        mt = payload_n_max / (n_total + (10 + 150 / 8) * n_aip)
    # unit bit / us or Mb/s
    return mt

def get_goodput(snr):
    opt_bcr = np.array([train_blong_bcr[extf][snr] for extf in extfs])
    opt_detr = np.array([train_dr[extf][snr] for extf in extfs])
    # print(opt_bcr)
    # print(opt_detr)
    _, opt_extf, _ = go.goodput_optimiation_nlos_fixn(opt_bcr, opt_detr, pdr_thres=0.2)
    if opt_extf is None:
        raise Exception("No suitable extf")
    print(snr, opt_extf)
    blong_th = max_throughput(opt_extf, 255) * test_blong_bcr[opt_extf][snr] * test_dr[opt_extf][snr]
    print("blong param", "MT", max_throughput(opt_extf, 255), "DR", test_dr[opt_extf][snr], "DR", test_blong_bcr[opt_extf][snr])
    stft_th = max_throughput(64, 255) * stft_dr[64][snr] * test_stft_bcr[64][snr]
    pd_th = max_throughput(1, 255) * test_dr[1][snr] * test_pd_bcr[1][snr]
    print("native", "MT", max_throughput(1, 255), "DR", test_dr[1][snr], "BCR", test_pd_bcr[1][snr])
    return blong_th, stft_th, pd_th

def get_throughput():
    snrs = np.arange(-18, -4)
    ths = []
    for snr in snrs:
        b, s, p = get_goodput(snr)
        ths.append([b, s, p])
    ths = np.array(ths)
    return ths


tx_power = 5 * 3 * 1e-3
data_len = 100 * 8  # 100B
ths = get_throughput() * 1e6
t = data_len / ths
energy = t * tx_power
print(energy)
print("stft", np.mean((energy[:, 0] - energy[:, 1]) / energy[:, 1]))
print("native", np.mean((energy[:, 0] - energy[:, 2]) / energy[:, 2]))

np.savetxt("./output/energy_consumption.csv", np.hstack([np.arange(-18, -4).reshape(-1, 1), energy]), delimiter=",")
        