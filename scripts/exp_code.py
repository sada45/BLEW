import sys
sys.path.append("./scripts")
import numpy as np
import config
import torch
import data_collection.BLong_preamble_detection as pd
import utils
import matplotlib.pyplot as plt
from scramble_table import *
import DANF.blong_dnn as dnn
import raw_data_preprocessing as rdp
import opt.goodput_opt as go

f = None

cur_data = None
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
power_of_2 = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
target_aa_bit = np.array([0,0,0,1,1,0,1,1,0,1,1,1,1,1,0,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1], dtype=np.uint8)
preamble_bits = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
header_bits = np.array([0,0,0,0,1,1,1,0], dtype=np.uint8)
len_bits = np.array([1,1,1,1,1,1,1,1], dtype=np.uint8)

indoor_time_diff = np.array([[-9, 46, -43, -14, -39, 0, 0, 0],
                             [-39, -11, 10, -14, -28, 39, 0, 0],
                             [-25, -26, -19, 24, -33, 0, -24, 0],
                             [19, -23, -36, -29, 20, 7, -30, 45],
                             [25, 16, -39, -13, -4, 24, 0, 25],
                             [-16, 13, 3, -27, -17, 12, 0, 32],
                             [-1, 16, 35, -23, 24, -14, 0, 8]])

def case_study_gt(ground_truth_file, extf):
    scenario = "outdoor"
    gt_data = np.load("./raw_data/case_study/" + scenario + "/gt/" + ground_truth_file + ".npz")
    gt_signal = gt_data["arr_0"].flatten()[np.newaxis, :int(128e4)]
    gt_signal = utils.filter(gt_signal)
    gt_time = gt_data["arr_1"][0, 0]
    if extf >= 8:
        p_idx, corr = pd.ground_turth(gt_signal, extf, full=False, time_int=40)
    else:
        p_idx, corr, pa_corr = pd.preamble_detection_arcphase(gt_signal, extf, cluster_size=3)
    
    name = ground_truth_file
    print(name)
    # plt.figure(figsize=(12, 12))
    # plt.subplot(211)
    # plt.plot(utils.get_phase_cum_1d(gt_signal[0, :]))
    # for i in range(len(p_idx)):
    #     plt.plot([p_idx[i], p_idx[i]], [0, 0.008])
    # plt.savefig("./figs/123.pdf")
    # plt.close()
    print(p_idx)
    p_idx = gt_time + p_idx[0] / config.sample_rate
    # print(p_idx[1:] - p_idx[0:-1])
    np.savez("./processed_data/gt/" + scenario + "/" + name + "_gt.npz", p_idx=p_idx, gt_time=gt_time)

def get_all_ground_truth():
    for d in [20]:
        for extf in [64]:
            if extf <= 16:
                file_name = "{}m_8c_{}e_{}n_0".format(d, extf, 265)
            else:
                file_name = "{}m_8c_{}e_{}n_0".format(d, extf, 100)
            # case_study_gt(file_name, extf)
            case_study_gt(file_name, extf)
    
# case_study/indoor/{}m_8c_{}e_{}n_0
time_cmp = {2: -0.03}
def case_study_get_time_diff(raw_file, extf, time_diff_thres=0.015, load=True):
    if extf <= 16:
        num = 265
    else:
        num = 100
    if extf > 1:
        data_bits = np.concatenate([target_aa_bit, header_bits, rand_bits])
        data_bits = data_bits[:(num-1)*8]
    else:
        data_bytes = np.concatenate([np.array([0x70, 0xff], dtype=np.uint8), scramble_table[0, :255]])
        data_bytes = data_bytes ^ scramble_table[8, :257]
        data_bytes = np.concatenate([data_bytes, np.array([224, 247, 242], dtype=np.uint8)])
        data_bits = utils.bytes_to_bits(data_bytes)
        data_bits = np.concatenate([target_aa_bit, data_bits])
        # print(data_bytes)
    if not load:
        ground_truth_file = raw_file
        name = ground_truth_file[ground_truth_file.find("case_study/")+len("case_study/"):]
        gt_data = np.load("./processed_data/gt/" + name + "_gt.npz")
        #  - 0.02906287792933027
        gt_p_time = gt_data["p_idx"] - 0.02906287792933027
        # print(gt_p_time[1:] - gt_p_time[0:-1])
        raw_data = np.load("./raw_data/" + raw_file + ".npz")
        raw_signal_ori = raw_data["arr_0"].flatten()[np.newaxis, int(0):int(128e4)]
        time_stamp = raw_data["arr_1"][0, 0]
        raw_signal = utils.filter(raw_signal_ori)
        # s_pc, s_corr = pd.preamble_detection_stft(raw_signal.copy(), 64, 2)
        # p = utils.get_phase(raw_signal)
        # p = p[:, 1:] - p[:, 0:-1]
        p = utils.get_phase_diff(raw_signal)
        pa = pd.get_preamble_aa_mask(extf)
        pa_corr = pd.cal_corrlation(p, pa)[0]
        p_diff = utils.get_phase_cum(raw_signal)
        print(np.max(pa_corr))
        # plt.figure(figsize=(12, 18))
        # plt.subplot(311)
        # plt.plot(pa_corr)
        # plt.subplot(312)
        # plt.plot(raw_signal[0].real)
        # plt.plot(raw_signal[0].imag)
        # plt.subplot(313)
        # plt.plot(p_diff[0])
        # plt.savefig("./figs/pa_corr.pdf")
        # plt.close()
        if extf < 64:
            thres = 95
        else: 
            thres = 98
        p_idx, corr, pa_corr = pd.preamble_detection(raw_signal, extf, thres_percent=thres)
        p_idx = p_idx[0]
        # print(p_idx)
        pa_corr = pa_corr[0]
        pa_corr = np.max(pa_corr, axis=1)
        p_time = p_idx / config.sample_rate + time_stamp
        p_time = p_time[:, np.newaxis]
        time_diff = np.abs(p_time - gt_p_time)
        time_diff_neg = p_time - gt_p_time
        with open("./output/time_corr/" + name + ".csv", "w") as wf:
            for i in range(len(gt_p_time)):
                # If the data receiving of the raw data is done, we terminate the processing
                if len(p_time) == 0:
                    return 
                if gt_p_time[i] > p_time[-1]:
                    break
                time_diff_seg = time_diff[:, i]
                td = time_diff_neg[np.argmin(time_diff_seg), i]*1000
                print(td)
                wf.write("{}\n".format(td))

def deal_outliers(pbcrs, bit_num):
    mean_bcrs = []
    for i in range(pbcrs.shape[0]):
        bcrs = pbcrs[i, :]
        q1, q3 = np.percentile(bcrs, [25, 75])
        iqr = 1.5 * (q3 - q1)
        upper = q3 + iqr
        lower = q1 - iqr
        valid_idx = np.where((bcrs<=upper)&(bcrs>=lower))[0]
        mean_bcr = np.mean(bcrs[valid_idx]) / bit_num
        mean_bcrs.append(mean_bcr)
    return mean_bcrs

def case_study_process(raw_file, extf, time_diff_thres=0.015, load=True, time_offset=0):
    if extf <= 16:
        num = 265
    else:
        num = 100
    if extf > 1:
        data_bits = np.concatenate([target_aa_bit, header_bits, rand_bits])
        data_bits = data_bits[:(num-1)*8]
    else:
        data_bytes = np.concatenate([np.array([0x70, 0xff], dtype=np.uint8), scramble_table[0, :255]])
        data_bytes = data_bytes ^ scramble_table[8, :257]
        data_bytes = np.concatenate([data_bytes, np.array([224, 247, 242], dtype=np.uint8)])
        data_bits = utils.bytes_to_bits(data_bytes)
        data_bits = np.concatenate([target_aa_bit, data_bits])
        # print(data_bytes)
    if not load:
        ground_truth_file = raw_file
        name = ground_truth_file[ground_truth_file.find("case_study/")+len("case_study/"):]
        gt_data = np.load("./processed_data/gt/" + name + "_gt.npz")
        #  - 0.02906287792933027
        gt_p_time = gt_data["p_idx"] - 0.02906287792933027 + time_offset / 1000
        # print(gt_p_time[1:] - gt_p_time[0:-1])
        raw_data = np.load("./raw_data/" + raw_file + ".npz")
        raw_signal_ori = raw_data["arr_0"].flatten()[np.newaxis, :]
        time_stamp = raw_data["arr_1"][0, 0]
        raw_signal = utils.filter(raw_signal_ori)
        thres = 98
        p_idx, corr, pa_corr = pd.preamble_detection_arcphase(raw_signal, extf, thres_percent=thres)
        p_idx = p_idx[0]
        # print(p_idx)
        pa_corr = pa_corr[0]
        pa_corr = np.max(pa_corr, axis=1)
        p_time = p_idx / config.sample_rate + time_stamp
        p_time = p_time[:, np.newaxis]
        time_diff = np.abs(p_time - gt_p_time)
        time_diff_neg = p_time - gt_p_time
        bound_flag = False
        lost_num = 0
        total_num = 0
        processed_p_idx = []
        for i in range(len(gt_p_time)):
            # If the data receiving of the raw data is done, we terminate the processing
            if len(p_time) == 0:
                    return np.zeros(3), np.zeros(3), 1
            if gt_p_time[i] > p_time[-1]:
                break
            time_diff_seg = time_diff[:, i]
            print(time_diff_neg[np.argmin(time_diff_seg), i]*1000)
            # If after the first raw packet, all packets are take into account
            if bound_flag:
                total_num += 1
            
            # If there is no preamble with the close time with the ground truth
            avi_idx = np.where(time_diff_seg <= time_diff_thres)[0]
            if len(avi_idx) == 0:
                if bound_flag:
                    lost_num += 1
                continue 
            elif not bound_flag:
                bound_flag = True
                total_num += 1
            avi_pa_corr = pa_corr[avi_idx]
            best_fit_idx = np.argmax(avi_pa_corr)
            processed_p_idx.append(p_idx[avi_idx[best_fit_idx]])
        if total_num == 0:
            plr = 1
        else:
            plr = lost_num / total_num
        print("Packet lose rate =", plr)
        # Then we start to slice the data bytes
        skip_len = (80 + 150) * config.sample_pre_symbol
        if extf > 1:
            bits_pre_pdu = int(2040 // extf)
        else:
            bits_pre_pdu = 265 * 8
        packets = []
        f_packets = []
        done_flag = False
        for i in range(len(processed_p_idx)):
            pkt = []
            f_pkt = []
            ptr = 8 * extf * config.sample_pre_symbol + processed_p_idx[i]
            for j in range(8, num*8):
                if j % bits_pre_pdu == 0:
                    ptr += skip_len
                n_ptr = ptr + extf * config.sample_pre_symbol
                sp = ptr - config.extra
                ep = n_ptr + config.extra
                if ep >= raw_signal_ori.shape[1]:
                    done_flag = True
                    break
                pkt.append(raw_signal_ori[0, sp:ep])
                f_pkt.append(raw_signal[0, sp:ep])
                ptr = n_ptr
            if done_flag:
                break
            packets.append(pkt)
            f_packets.append(f_pkt)
        packets = np.array(packets)
        f_packets = np.array(f_packets) 
        np.savez("./processed_data/" + raw_file + ".npz", f_packets=f_packets, packets=packets, lost_num=lost_num, total_num=total_num)
    else:
        data = np.load("./processed_data/" + raw_file + ".npz")
        # f_packets = data["f_packets"]
        packets = data["packets"]
        lost_num = data["lost_num"]
        total_num = data["total_num"]
    if total_num == 0:
        return np.zeros(3), np.zeros(3), 1
    packets_shape = packets.shape
    packets = packets.reshape([-1, packets.shape[2]])
    packets = utils.signal_normalize(packets, 1)
    f_packets = utils.filter(packets).astype(np.complex64)
    packets = packets.reshape(packets_shape)
    f_packets = f_packets.reshape(packets_shape)
    packets = packets[:, :, config.extra:packets.shape[2]-config.extra]
    f_packets = f_packets[:, :, config.extra:f_packets.shape[2]-config.extra]
    cpns, bcrs, pbcrs = dnn.dnn_decode(packets, f_packets, extf, data_bits[:(num-1)*8])
    total_bits = packets.shape[0] * packets.shape[1]
    if extf > 1:
        bcrs = bcrs / total_bits
    else:
        bcrs = deal_outliers(pbcrs, packets.shape[1])
    cpns = cpns / packets.shape[0]
    print(raw_file, extf, "bcr:", bcrs, "cpns:", cpns)
    return bcrs, cpns, lost_num / total_num

def wifi_int_data_save():
    extf = 64
    raw_signal = np.load("/liymdata/liym/BLong/raw_data/case_study/gt/0m_8c_64e_100n_2.npz")["arr_0"].flatten()[np.newaxis, :]
    noise = np.load("/liymdata/liym/BLong/raw_data/case_study/wifi/5m_4c_40M_0.npz")["arr_0"].flatten()[np.newaxis, :]
    cut_len = np.min([raw_signal.shape[1], noise.shape[1]])
    raw_signal = raw_signal[:, :cut_len]
    noise = noise[:, :cut_len]
    
    filtered_signal = utils.filter(raw_signal)
    p_idx, _ = pd.ground_turth(raw_signal, extf, False)
    data_bits = np.concatenate([preamble_bits, target_aa_bit, header_bits, rand_bits])
    for dextf in [1, 2, 4, 8, 16, 32]:
        dext_syms = [[], []]
        dext_labels = [[], []]
        dext_snrs = [[], []]
        ext_syms = [[], []]
        ext_labels = [[], []]
        ext_snrs = [[], []]
        for snr in range(-30, 1, 1):
            int_signal = utils.wifi_awgn(filtered_signal[0, :], snr, extf, p_idx[0], noise[0, :])
            sym_0, sym_1, _, _, _, _ = rdp.symbol_slice(int_signal, filtered_signal[0, :], noise[0, :], p_idx[0], extf, data_bits[:100], extra=config.extra)
            syms = np.vstack([sym_0, sym_1])
            labels = np.concatenate([np.zeros(sym_0.shape[0], dtype=np.uint8), np.ones(sym_1.shape[0], dtype=np.uint8)])
            dext_sym, _, _, dext_label = rdp.symbol_dext(syms, None, None, labels, extf, dextf)
            num = int(syms.shape[0] / 10)
            shuffle_idx = np.random.permutation(dext_sym.shape[0])
            shuffle_train_idx = shuffle_idx[:num]
            shuffle_test_idx = shuffle_idx[num:2*num]
            ext_shuffle_idx = np.random.permutation(syms.shape[0])
            print(ext_shuffle_idx[:10])
            ext_shuffle_train_idx = ext_shuffle_idx[:num]
            ext_shuffle_test_idx = ext_shuffle_idx[num:2*num]
            print(snr, dextf, num)

            dext_syms[0].append(dext_sym[shuffle_train_idx, :])
            dext_syms[1].append(dext_sym[shuffle_test_idx, :])
            dext_labels[0].append(dext_label[shuffle_train_idx])
            dext_labels[1].append(dext_label[shuffle_test_idx])
            s1 = np.array([snr for _ in range(num)])
            dext_snrs[0].append(s1)
            dext_snrs[1].append(s1)

            ext_syms[0].append(syms[ext_shuffle_train_idx, :])
            ext_syms[1].append(syms[ext_shuffle_test_idx, :])
            ext_labels[0].append(labels[ext_shuffle_train_idx])
            ext_labels[1].append(labels[ext_shuffle_test_idx])
            ext_snrs[0].append(s1)
            ext_snrs[1].append(s1)
        # dext_syms = np.vstack(dext_syms)
        # dext_labels = np.concatenate(dext_labels)
        # dext_snrs = np.concatenate(dext_snrs)
        # ext_syms = np.vstack(ext_syms)
        # ext_labels = np.concatenate(ext_labels)
        # ext_snrs = np.concatenate(ext_snrs)
        print(dextf)
        np.savez("processed_data/wifi_intt_new_{}e_train.npz".format(dextf), sym=np.vstack(dext_syms[0]), label=np.concatenate(dext_labels[0]), snr=np.concatenate(dext_snrs[0]))
        np.savez("processed_data/wifi_intt_new_{}e_test.npz".format(dextf), sym=np.vstack(dext_syms[1]), label=np.concatenate(dext_labels[1]), snr=np.concatenate(dext_snrs[1]))
        if dextf == 1:
            np.savez("processed_data/wifi_int_new_{}e_train.npz".format(extf), sym=np.vstack(ext_syms[0]), label=np.concatenate(ext_labels[0]), snr=np.concatenate(ext_snrs[0]))
            np.savez("processed_data/wifi_int_new_{}e_test.npz".format(extf), sym=np.vstack(ext_syms[1]), label=np.concatenate(ext_labels[1]), snr=np.concatenate(ext_snrs[1]))

def wifi_int_data_save_sensys():
    extf = 64
    raw_signal = np.load("/liymdata/liym/BLong/raw_data/case_study/gt/0m_8c_64e_100n_2.npz")["arr_0"].flatten()[np.newaxis, :]
    noise = np.load("/liymdata/liym/BLong/raw_data/case_study/wifi/5m_4c_40M_0.npz")["arr_0"].flatten()[np.newaxis, :]
    cut_len = np.min([raw_signal.shape[1], noise.shape[1]])
    raw_signal = raw_signal[:, :cut_len]
    noise = noise[:, :cut_len]
    
    filtered_signal = utils.filter(raw_signal)
    p_idx, _ = pd.ground_turth(raw_signal, extf, False)
    data_bits = np.concatenate([preamble_bits, target_aa_bit, header_bits, rand_bits])
    for dextf in [1, 2, 4, 8, 16, 32]:
        dext_syms = [[], []]
        dext_labels = [[], []]
        dext_snrs = [[], []]
        ext_syms = [[], []]
        ext_labels = [[], []]
        ext_snrs = [[], []]
        shuffle_idx = None
        ext_shuffle_idx = None
        for snr in range(-30, 1, 1):
            int_signal = utils.wifi_awgn(filtered_signal[0, :], snr, extf, p_idx[0], noise[0, :])
            sym_0, sym_1, _, _, _, _ = rdp.symbol_slice(int_signal, filtered_signal[0, :], noise[0, :], p_idx[0], extf, data_bits[:100], extra=config.extra)
            syms = np.vstack([sym_0, sym_1])
            labels = np.concatenate([np.zeros(sym_0.shape[0], dtype=np.uint8), np.ones(sym_1.shape[0], dtype=np.uint8)])
            dext_sym, _, _, dext_label = rdp.symbol_dext(syms, None, None, labels, extf, dextf)
            num = int(syms.shape[0] / 10)
            if shuffle_idx is None:
                shuffle_idx = np.random.permutation(dext_sym.shape[0])
            shuffle_train_idx = shuffle_idx[:num]
            shuffle_test_idx = shuffle_idx[num:2*num]
            if ext_shuffle_idx is None:
                ext_shuffle_idx = np.random.permutation(syms.shape[0])
            print(ext_shuffle_idx[:10])
            ext_shuffle_train_idx = ext_shuffle_idx[:num]
            ext_shuffle_test_idx = ext_shuffle_idx[num:2*num]
            print(snr, dextf, num)

            dext_syms[0].append(dext_sym[shuffle_train_idx, :])
            dext_syms[1].append(dext_sym[shuffle_test_idx, :])
            dext_labels[0].append(dext_label[shuffle_train_idx])
            dext_labels[1].append(dext_label[shuffle_test_idx])
            s1 = np.array([snr for _ in range(num)])
            dext_snrs[0].append(s1)
            dext_snrs[1].append(s1)

            ext_syms[0].append(syms[ext_shuffle_train_idx, :])
            ext_syms[1].append(syms[ext_shuffle_test_idx, :])
            ext_labels[0].append(labels[ext_shuffle_train_idx])
            ext_labels[1].append(labels[ext_shuffle_test_idx])
            ext_snrs[0].append(s1)
            ext_snrs[1].append(s1)
        # dext_syms = np.vstack(dext_syms)
        # dext_labels = np.concatenate(dext_labels)
        # dext_snrs = np.concatenate(dext_snrs)
        # ext_syms = np.vstack(ext_syms)
        # ext_labels = np.concatenate(ext_labels)
        # ext_snrs = np.concatenate(ext_snrs)
        print(dextf)
        np.savez("processed_data/wifi_int_new_{}e_train.npz".format(dextf), sym=np.vstack(dext_syms[0]), label=np.concatenate(dext_labels[0]), snr=np.concatenate(dext_snrs[0]))
        np.savez("processed_data/wifi_int_new_{}e_test.npz".format(dextf), sym=np.vstack(dext_syms[1]), label=np.concatenate(dext_labels[1]), snr=np.concatenate(dext_snrs[1]))
        if dextf == 1:
            np.savez("processed_data/wifi_int_new_{}e_train.npz".format(extf), sym=np.vstack(ext_syms[0]), label=np.concatenate(ext_labels[0]), snr=np.concatenate(ext_snrs[0]))
            np.savez("processed_data/wifi_int_new_{}e_test.npz".format(extf), sym=np.vstack(ext_syms[1]), label=np.concatenate(ext_labels[1]), snr=np.concatenate(ext_snrs[1]))



def wifi_int_exp(dextf=8):
    with open("./output/wifi_interference.log", 'w') as f:
        extf = 64
        raw_signal = np.load("/data/liym/BLong/raw_data/case_study/gt/0m_8c_64e_100n_2.npz")["arr_0"].flatten()[np.newaxis, :]
        noise = np.load("/data/liym/BLong/raw_data/case_study/wifi/5m_4c_40M_0.npz")["arr_0"].flatten()[np.newaxis, :]
        cut_len = np.min([raw_signal.shape[1], noise.shape[1]])
        raw_signal = raw_signal[:, :cut_len]
        noise = noise[:, :cut_len]
        
        filtered_signal = utils.filter(raw_signal)
        p_idx, _ = pd.ground_turth(raw_signal, extf, False)
        data_bits = np.concatenate([preamble_bits, target_aa_bit, header_bits, rand_bits])
        for snr in range(-20, 1, 1):
            int_signal = utils.wifi_awgn(filtered_signal[0, :], snr, extf, p_idx[0], noise[0, :])
            sym_0, sym_1, _, _, _, _ = rdp.symbol_slice(int_signal, filtered_signal[0, :], noise[0, :], p_idx[0], extf, data_bits[:100], extra=config.extra)
            syms = np.vstack([sym_0, sym_1])
            labels = np.concatenate([np.zeros(sym_0.shape[0], dtype=np.uint8), np.ones(sym_1.shape[0], dtype=np.uint8)])
            bits, correct_num = dnn.decode_ber(syms, labels, extf)
            ber = (syms.shape[0] - correct_num) / syms.shape[0]
            s = "snr={}, extf={}, dnn={}, stft={}, pd={},".format(snr, extf, ber[0], ber[1], ber[2])
            print(s, end="")
            f.write(s)
            dext_syms, _, _, dext_labels = rdp.symbol_dext(syms, None, None, labels, extf, dextf)
            dext_bits, dext_correct_num = dnn.decode_ber(dext_syms, dext_labels, dextf)
            ber = (dext_syms.shape[0] - dext_correct_num) / dext_syms.shape[0]
            s = "dextf={}, dext_dnn={}, dext_stft={}, dext_pd={}".format(extf, ber[0], ber[1], ber[2])
            print(s)
            f.write(s + "\n")

def time_diff_cmp(ground_truth_file):
    name = ground_truth_file[ground_truth_file.find("case_study/")+len("case_study/"):]
    gt_data = np.load("./processed_data/gt/" + name + "_gt.npz")
    gt_p_time = gt_data["p_idx"] - 0.02906287792933027
    print(gt_p_time[1:] - gt_p_time[0:-1])

def goodput_exp(scenario, n_max=None, extf=None, native=0):
    if scenario == "indoor":
        dis = [5, 10, 15, 20, 25, 30, 40]
    elif scenario == "outdoor":
        dis = [10, 20, 30, 40, 60, 80, 100]
    else:
        raise Exception("No such as scenario")
    if n_max is None or extf is None:
        update_flag = True
    else:
        update_flag = False
    for d in dis:
        if update_flag:
            n_max, extf, _ = go.goodput_optimiation_distance(d, scenario)

        if extf >= 32:
            num = 100
        else:
            num = 265
        if extf > 1:
            data_bits = np.concatenate([target_aa_bit, header_bits, rand_bits])
            data_bits = data_bits[:(num-1)*8]
        else:
            # The 0x70 is the header, the 0xff is the length of the header, which is set to 255 (maximum)
            data_bytes = np.concatenate([np.array([0x70, 0xff], dtype=np.uint8), scramble_table[0, :255]])
            data_bytes = data_bytes ^ scramble_table[8, :257]
            data_bytes = np.concatenate([data_bytes, np.array([224, 247, 242], dtype=np.uint8)])
            data_bits = utils.bytes_to_bits(data_bytes)
            data_bits = np.concatenate([target_aa_bit, data_bits])
        if extf >= 32:
            f_name = "/{}m_8c_{}e_{}n_0.npz".format(d, extf, 100)
        else:
            f_name = "/{}m_8c_{}e_{}n_0.npz".format(d, extf, 265)
        data = np.load("./processed_data/case_study/" + scenario + f_name)
        packets = data["packets"]
        packet_num = packets.shape[0]
        gt_bits = np.concatenate([data_bits for _ in range(packet_num)])
        packets = packets.reshape([-1, packets.shape[2]])
        bits, correct_num = dnn.decode_ber(packets, gt_bits, extf)
        packet_len = (n_max + 10) * 8
        if native == 0:
            if extf == 1:
                bits = bits[1]
            else:
                bits = bits[0]
        elif native == 1:
            bits = bits[1]
        else:
            bits = bits[2]
        cut_len = len(bits) - len(bits) % packet_len
        bits = bits[:cut_len]
        bits = bits.reshape([-1, packet_len])
        gt_bits = gt_bits[:cut_len].reshape([-1, packet_len])
        decode_corr = np.sum(np.equal(bits, gt_bits), axis=1)
        pdr = len(np.where(decode_corr==packet_len)[0]) / len(decode_corr)
        print("{}, {}, {}, {}".format(d, n_max, extf, pdr))

def case_study_all():
    scenario = "indoor"
    with open("./output/bcrs.log", "a") as f: 
        for extf in [1]:
            for d in [5, 10, 15, 20, 25, 30, 40]:
                if scenario == "indoor":
                    tf = indoor_time_diff[int(np.log2(extf)), int(d // 5)-1]
                else:
                    tf = indoor_time_diff[int(np.log2(extf)), int(d // 10)-1]
                if extf > 16:
                    bcrs, cpns, plr = case_study_process("case_study/" + scenario + "/{}m_8c_{}e_{}n_0".format(d, extf, 100), extf, load=False, time_offset=tf)
                else:
                    bcrs, cpns, plr = case_study_process("case_study/" + scenario + "/{}m_8c_{}e_{}n_0".format(d, extf, 265), extf, load=False, time_offset=tf)
                s = "{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(d, extf, plr, bcrs[0], bcrs[1], bcrs[2], cpns[0], cpns[1], cpns[2])
                f.write(s)
                f.flush()

def case_study_time_diff_all(scenario):
    for extf in [64]:
        # utils.power_of_2
        # [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        for d in [20, 30, 40, 50, 60, 70, 80, 90]:
            if extf > 16:
                case_study_get_time_diff("case_study/" + scenario + "/{}m_8c_{}e_{}n_0".format(d, extf, 100), extf, load=False)
            else:
                case_study_get_time_diff("case_study/" + scenario + "/{}m_8c_{}e_{}n_0".format(d, extf, 265), extf, load=False)

if __name__ == "__main__":
    # goodput_exp("indoor", native=1)
    # time_diff_cmp("case_study/indoor/{}m_8c_{}e_{}n_0".format(5, 64, 100))
    # get_all_ground_truth()
    # wifi_int_exp()
    # case_study_gt("case_study/indoor/gt/5m_8c_1e_265n_0", 1)
    # case_study_gt("case_study/gt/0m_8c_64e_100n_2", 64)
    # case_study_gt("case_study/gt/0m_8c_64e_100n_3", 64)
    wifi_int_data_save_sensys() 
    # time_diff_cmp("case_study/indoor/{}m_8c_{}e_{}n_0".format(35, 32, 100))
    # goodput_exp("indoor")
    # get_all_ground_truth()
    # case_study_time_diff_all("outdoor")
    # case_study_gt("20m_8c_64e_100n_2", 64)
    # case_study_get_time_diff("case_study/" + "outdoor" + "/{}m_8c_{}e_{}n_0".format(80, 64, 100), 64, load=False)
    # case_study_gt("{}m_8c_{}e_{}n_0".format(20, 64, 100), 64)
    # wifi_int_data_save()
