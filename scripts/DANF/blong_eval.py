import numpy as np
import torch
from blong_nn_components import *
import ble_data_loader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import raw_data_processing
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def symbol_stft_decoding(spec, rem_num):
    bit_data = torch.zeros(spec.shape[0], dtype=torch.uint8)
    spec = spec[:, 0, :, :] + spec[:, 1, :, :]
    high_freq = torch.sum(spec[:, :rem_num, :], dim=(1, 2))
    low_freq = torch.sum(spec[:, rem_num: 2*rem_num], dim=(1, 2))
    freq_diff = high_freq - low_freq
    bit_1_idx = torch.where(freq_diff > 0)[0]
    bit_data[bit_1_idx] = 1
    return bit_data


def model_eval_chan(blong_fe, blong_nf, cnn_fe, cnn_nf, test_data_loader, native=False):
    blong_fe.eval()
    blong_nf.eval()
    cnn_fe.eval()
    cnn_nf.eval()
    freq_resolution = config.sample_rate / (config.stft_window_size * config.sample_pre_symbol)
    rem_num = int(1e6 // freq_resolution)
    with torch.no_grad():
        blong_test_labels = torch.zeros(len(test_data_loader.dataset), dtype=torch.uint8)
        cnn_test_labels = torch.zeros(len(test_data_loader.dataset), dtype=torch.uint8)
        stft_test_labels = torch.zeros(len(test_data_loader.dataset), dtype=torch.uint8)
        ground_truth_labels = torch.zeros(len(test_data_loader.dataset), dtype=torch.uint8)
        chan_lables = torch.zeros(len(test_data_loader.dataset), dtype=torch.int64)
        ptr = 0
        for idx, (noise_sym, sym, label, chan_label, db) in enumerate(test_data_loader):
            noise_sym = noise_sym.to(device)
            label = label.to(device)
            # Neural-enhance noise filter
            blong_denoise_spec = blong_nf(blong_fe(noise_sym))
            blong_denoise_spec = blong_denoise_spec * noise_sym
            cnn_denoise_spec = cnn_nf(cnn_fe(noise_sym))
            cnn_denoise_spec = cnn_denoise_spec * noise_sym
            # Symbol decoding
            blong_test_labels[ptr: ptr+noise_sym.shape[0]] = symbol_stft_decoding(blong_denoise_spec, rem_num)
            cnn_test_labels[ptr: ptr+noise_sym.shape[0]] = symbol_stft_decoding(cnn_denoise_spec, rem_num)
            if native:
                stft_test_labels[ptr: ptr+noise_sym.shape[0]] = symbol_stft_decoding(noise_sym, rem_num)
            ground_truth_labels[ptr: ptr+noise_sym.shape[0]] = label.type(torch.uint8)
            chan_lables[ptr:ptr+noise_sym.shape[0]] = chan_label.type(torch.int64)
            ptr += noise_sym.shape[0]
        # s = (prefix + "epoch={}, ber={}").format(cur_epoch, (len(test_data_loader.dataset)-correct_num) / len(test_data_loader.dataset))
        # f.write(s + "\n")
        # f.flush()
        blong_chan_ber = torch.zeros(40, dtype=torch.float32)
        cnn_chan_ber = torch.zeros(40, dtype=torch.float32)
        stft_chan_ber = torch.zeros(40, dtype=torch.float32)
        for i in range(40):
            idx = torch.where(chan_lables==i)
            blong_t_labels = blong_test_labels[idx]
            cnn_t_labels = cnn_test_labels[idx]
            stft_t_labels = stft_test_labels[idx]
            gt_labels = ground_truth_labels[idx]
            blong_chan_ber[i] = torch.where(blong_t_labels==gt_labels)[0].shape[0] / idx[0].shape[0]
            cnn_chan_ber[i] = torch.where(cnn_t_labels==gt_labels)[0].shape[0] / idx[0].shape[0]
            stft_chan_ber[i] = torch.where(stft_t_labels==gt_labels)[0].shape[0] / idx[0].shape[0]

    return blong_chan_ber, cnn_chan_ber, stft_chan_ber

def model_eval(blong_fe, blong_nf, cnn_fe, cnn_nf, test_data_loader, native=False):
    blong_fe.eval()
    blong_nf.eval()
    cnn_fe.eval()
    cnn_nf.eval()
    freq_resolution = config.sample_rate / (config.stft_window_size * config.sample_pre_symbol)
    rem_num = int(1e6 // freq_resolution)
    with torch.no_grad():
        blong_test_labels = torch.zeros(len(test_data_loader.dataset), dtype=torch.uint8)
        cnn_test_labels = torch.zeros(len(test_data_loader.dataset), dtype=torch.uint8)
        stft_test_labels = torch.zeros(len(test_data_loader.dataset), dtype=torch.uint8)
        ground_truth_labels = torch.zeros(len(test_data_loader.dataset), dtype=torch.uint8)
        ptr = 0
        for idx, (noise_sym, _, label, _, _) in enumerate(test_data_loader):
            noise_sym = noise_sym.to(device)
            label = label.to(device)
            # Neural-enhance noise filter
            blong_denoise_spec = blong_nf(blong_fe(noise_sym))
            blong_denoise_spec = blong_denoise_spec * noise_sym
            cnn_denoise_spec = cnn_nf(cnn_fe(noise_sym))
            cnn_denoise_spec = cnn_denoise_spec * noise_sym
            # Symbol decoding
            blong_test_labels[ptr: ptr+noise_sym.shape[0]] = symbol_stft_decoding(blong_denoise_spec, rem_num)
            cnn_test_labels[ptr: ptr+noise_sym.shape[0]] = symbol_stft_decoding(cnn_denoise_spec, rem_num)
            if native:
                stft_test_labels[ptr: ptr+noise_sym.shape[0]] = symbol_stft_decoding(noise_sym, rem_num)
            ground_truth_labels[ptr: ptr+noise_sym.shape[0]] = label.type(torch.uint8)
            ptr += noise_sym.shape[0]
        # s = (prefix + "epoch={}, ber={}").format(cur_epoch, (len(test_data_loader.dataset)-correct_num) / len(test_data_loader.dataset))
        # f.write(s + "\n")
        # f.flush()
        blong_bcr = torch.where(blong_test_labels==ground_truth_labels)[0].shape[0] / len(test_data_loader.dataset)
        cnn_bcr = torch.where(cnn_test_labels==ground_truth_labels)[0].shape[0] / len(test_data_loader.dataset)
        if native:
            stft_bcr = torch.where(stft_test_labels==ground_truth_labels)[0].shape[0] / len(test_data_loader.dataset)
            return blong_bcr, cnn_bcr, stft_bcr
        else:
            return blong_bcr, cnn_bcr

def native_stft_eval(extf):
    freq_resolution = config.sample_rate / (config.stft_window_size * config.sample_pre_symbol)
    rem_num = int(1e6 // freq_resolution)
    bcr = np.zeros(41)
    for snr in range(-30, 11, 1):
        test_dataset = ble_data_loader.ble_torch_dataset(extf, True, snr)
        test_data_loader = DataLoader(test_dataset, 64, shuffle=False, num_workers=10)
        correct_num = 0
        total_num = 0
        for idx, (noise_sym, sym, label, chan_label, db) in enumerate(test_data_loader):
            noise_sym = noise_sym.to(device)
            decode_label = symbol_stft_decoding(noise_sym, rem_num)
            correct_num += torch.where(decode_label==label)[0].shape[0]
            total_num += noise_sym.shape[0]
        bcr[snr+30] = correct_num / total_num
        print(extf, snr, bcr[snr+30])
    with open("./logs/native_stft_without_hanning_extf=" + str(extf) + ".txt", 'w') as f:
        for snr in range(-30, 11, 1):
            f.write("{}={}\n".format(snr, bcr[snr+30]))
        f.flush()
    
def native_fft_eval(extf):
    bcr = np.zeros(41)
    for snr in range(-30, 11, 1):
        test_dataset = ble_data_loader.ble_raw_dataset(extf, True, snr)
        test_data_loader = DataLoader(test_dataset, 128, shuffle=False, num_workers=10)
        correct_num = 0
        total_num = 0
        half_bin_num = int(extf*config.sample_pre_symbol/2)
        for idx, (noise_signal, signal, label, chan_label, db) in enumerate(test_data_loader):
            mean_val = torch.mean(noise_signal, dim=1).unsqueeze(1)
            noise_signal = noise_signal - mean_val
            decode_label = torch.zeros(noise_signal.shape[0], dtype=torch.uint8)
            fft_res = torch.fft.fft(noise_signal, dim=1)
            fft_res = torch.abs(fft_res)
            high_freq = torch.sum(fft_res[:, :half_bin_num], dim=1)
            low_freq = torch.sum(fft_res[:, half_bin_num: ], dim=1)
            decode_label[torch.where(high_freq>low_freq)] = 1
            correct_num += torch.where(decode_label==label)[0].shape[0]
            total_num += noise_signal.shape[0]
        bcr[snr+30] = correct_num / total_num
        print(snr, bcr[snr+30])
    with open("./logs/native_stft_extf=" + str(extf) + ".txt", 'w') as f:
        for snr in range(-30, 11, 1):
            f.write("{}={}\n".format(snr, bcr[snr+30]))
        f.flush()

batch_size = 64
def blong_eval(extf, round):
    w_mode = "a"
    if round == 0:
        w_mode = "w"
    with open("./logs/eval_test_extf=" + str(extf) + ".txt", w_mode) as f:
        models = os.listdir("./models/")
        blong_models_list = []
        cnn_models_list = []
        for model in models:
            if model.find("blong_nn_extf=" + str(extf) + "_round=" + str(round) + "_final") != -1 and model.find("copy") == -1:
                blong_models_list.append(model)
            elif model.find("cnn_extf=" + str(extf) + "_round=" + str(round) + "_final") != -1 and model.find("copy") == -1:
                cnn_models_list.append(model)
        num_models = min(len(blong_models_list), len(cnn_models_list))
        max_num_models = max(len(blong_models_list), len(cnn_models_list))
        remove_item = []
        remove_list = None
        if len(blong_models_list) > len(cnn_models_list):
            remove_list = blong_models_list
        else:
            remove_list = cnn_models_list
        for model in remove_list:
            if model.find("final_") != -1:
                model_idx = int(model[model.find("final_")+6:model.find(".pt")])
                if model_idx < max_num_models - num_models:
                    remove_item.append(model)
        for r in remove_item:
            remove_list.remove(r)


        native_chan_bers = torch.zeros(41)
        blong_chan_bers = torch.zeros(num_models, 41)
        cnn_chan_bers = torch.zeros(num_models, 41)
        for i in range(len(blong_models_list)):
            f.write(blong_models_list[i] + "----")
            blong_model = torch.load("./models/" + blong_models_list[i])
            blong_fe = FeatureExtractor().to(device)
            blong_nf = NoiseFilter(extf).to(device)
            blong_fe.load_state_dict(blong_model["fe_state"])
            blong_nf.load_state_dict(blong_model["nf_state"])
            f.write(cnn_models_list[i] + "\n")
            cnn_model = torch.load("./models/" + cnn_models_list[i])
            cnn_fe = FeatureExtractor().to(device)
            cnn_nf = NoiseFilter(extf).to(device)
            cnn_fe.load_state_dict(cnn_model["fe_state"])
            cnn_nf.load_state_dict(cnn_model["nf_state"])
    
            for snr in range(-30, 11, 1):
                test_data_set = ble_data_loader.ble_torch_dataset(extf, False, snr)
                data_loader = DataLoader(test_data_set, batch_size, shuffle=False, num_workers=10)
                if i == 0 and round == 0:
                    blong_chan_ber, cnn_chan_ber, stft_chan_ber = model_eval(blong_fe, blong_nf, cnn_fe, cnn_nf, data_loader, True)
                    native_chan_bers[snr+30] = stft_chan_ber
                else:
                    blong_chan_ber, cnn_chan_ber = model_eval(blong_fe, blong_nf, cnn_fe, cnn_nf, data_loader)
                blong_chan_bers[i, snr+30] = blong_chan_ber
                cnn_chan_bers[i, snr+30] = cnn_chan_ber
                f.write("{}={},{},{}\n".format(snr, blong_chan_bers[i, snr+30], cnn_chan_bers[i, snr+30], native_chan_bers[snr+30]))
                f.flush()
            # f.write("=============================================\n")

        # total_blong_bcr = torch.sum(blong_chan_bers, dim=1)
        # total_cnn_bcr = torch.sum(cnn_chan_bers, dim=1)
        # blong_max_idx = torch.argmax(total_blong_bcr)
        # cnn_max_idx = torch.argmax(total_cnn_bcr)
        # for i in range(41):
        #     f.write("{}={},{},{}\n".format(i-30, blong_chan_bers[blong_max_idx, i], cnn_chan_bers[cnn_max_idx, i], native_chan_bers[i]))
        #     f.flush()
            # f.write("blong")
            # for i in range(40):
            #     f.write("{}, ".format(blong_chan_ber[i]))
            # f.write("\n")
            # for i in range(40):
            #     f.write("{},".format(cnn_chan_ber[i]))
            # f.write("\n")
            # f.flush()

def blong_eval_single(extf):
    with open("./logs/eval_test_extf=" + str(extf) + ".txt", 'w') as f:
        native_chan_bers = torch.zeros(41)
        blong_chan_bers = torch.zeros(41)
        cnn_chan_bers = torch.zeros(41)
        blong_model = torch.load("./models/blong_nn_extf=" + str(extf) + "_final.pt")
        blong_fe = FeatureExtractor().to(device)
        blong_nf = NoiseFilter(extf).to(device)
        blong_fe.load_state_dict(blong_model["fe_state"])
        blong_nf.load_state_dict(blong_model["nf_state"])
        cnn_model = torch.load("./models/cnn_extf=" + str(extf) + "_final.pt")
        cnn_fe = FeatureExtractor().to(device)
        cnn_nf = NoiseFilter(extf).to(device)
        cnn_fe.load_state_dict(cnn_model["fe_state"])
        cnn_nf.load_state_dict(cnn_model["nf_state"])

        for snr in range(-30, 11, 1):
            test_data_set = ble_data_loader.ble_torch_dataset(extf, False, snr)
            data_loader = DataLoader(test_data_set, batch_size, shuffle=False, num_workers=10)
            blong_chan_ber, cnn_chan_ber, stft_chan_ber = model_eval(blong_fe, blong_nf, cnn_fe, cnn_nf, data_loader, True)
            native_chan_bers[snr+30] = stft_chan_ber
            blong_chan_bers[snr+30] = blong_chan_ber
            cnn_chan_bers[snr+30] = cnn_chan_ber
            f.write("{}={},{},{}\n".format(snr, blong_chan_bers[snr+30], cnn_chan_bers[snr+30], native_chan_bers[snr+30]))
            f.flush()

def eval_all(extf):
    with open("./logs/eval_test_extf=" + str(extf) + ".txt", 'r+') as f:
        lines = f.readlines()
        blong_model_list = []
        cnn_model_list = []
        blong_acc_list = []
        cnn_acc_list = []
        stft_acc = np.zeros(41, dtype=np.float64)
        cur_ptr = 0

        for line in lines:
            idx = line.find("----")
            if idx != -1:
                blong_model_list.append(line[:idx])
                cnn_model_list.append(line[idx+4:-1])
                blong_acc_list.append(np.zeros(41, dtype=np.float64))
                cnn_acc_list.append(np.zeros(41, dtype=np.float64))
                cur_ptr = 0
            else:
                acc_line = line[line.find("=")+1: ]
                accs = acc_line.split(",")
                blong_acc_list[-1][cur_ptr] = float(accs[0])
                cnn_acc_list[-1][cur_ptr] = float(accs[1])
                if float(accs[2]) > 0:
                    stft_acc[cur_ptr] = float(accs[2])
                cur_ptr += 1
        blong_acc_list = np.array(blong_acc_list)
        cnn_acc_list = np.array(cnn_acc_list)
        blong_acc_sum = np.sum(blong_acc_list, axis=1)
        cnn_acc_sum = np.sum(cnn_acc_list, axis=1)
        blong_max_idx = np.argmax(blong_acc_sum)
        cnn_max_idx = np.argmax(cnn_acc_sum)

        f.seek(0, 2)
        f.write("\n===================\n")
        for i in range(41):
            f.write("{}={},{},{}\n".format(i-30, blong_acc_list[blong_max_idx, i], cnn_acc_list[cnn_max_idx, i], stft_acc[i]))
        f.write(blong_model_list[blong_max_idx] + ",")
        f.write(cnn_model_list[cnn_max_idx] + "\n")
        f.flush()
            




def eval_source_target(extf):
    with open("./logs/eval_test_extf=" + str(extf) + ".txt", 'r') as f:
        lines = f.readlines()
        line = lines[-1]
        best_model_name = line.split(",")[1][:-1]  # Takse the DNN 
    
    blong_model = torch.load("./models/" + best_model_name)
    blong_fe = FeatureExtractor().to(device)
    blong_nf = NoiseFilter(extf).to(device)
    blong_fe.load_state_dict(blong_model["fe_state"])
    blong_nf.load_state_dict(blong_model["nf_state"])
    blong_fe.eval()
    blong_nf.eval()
    freq_resolution = config.sample_rate / (config.stft_window_size * config.sample_pre_symbol)
    rem_num = int(1e6 // freq_resolution)
    logf = open("./logs/source_target_extf=" + str(extf) + ".log", 'w')
    with torch.no_grad():
        for snr in range(-30, 11, 1):
            train_dataset = ble_data_loader.ble_torch_dataset(extf, True, snr)
            train_data_loader = DataLoader(train_dataset, 64, shuffle=False, num_workers=8)
            source_correct_num = 0
            for idx, (noise_sym, sym, label, chan_label, db) in enumerate(train_data_loader):
                noise_sym = noise_sym.to(device)
                denoise_sym = blong_nf(blong_fe(noise_sym)) * noise_sym
                decode_bits = symbol_stft_decoding(denoise_sym, rem_num)
                source_correct_num += np.where(decode_bits==label)[0].shape[0]
            del train_data_loader
            test_dataset = ble_data_loader.ble_torch_dataset(extf, False, snr)
            test_data_loader = DataLoader(test_dataset, 64, shuffle=False, num_workers=8)
            target_correct_num = 0
            for idx, (noise_sym, sym, label, chan_label, db) in enumerate(test_data_loader):
                noise_sym = noise_sym.to(device)
                denoise_sym = blong_nf(blong_fe(noise_sym)) * noise_sym
                decode_bits = symbol_stft_decoding(denoise_sym, rem_num)
                target_correct_num += np.where(decode_bits==label)[0].shape[0]
            # We store the BER
            source_ber = (len(train_dataset) - source_correct_num) / len(train_dataset)
            target_ber = (len(test_dataset) - target_correct_num) / len(test_dataset)
            logf.write("{},{},{},{}\n".format(snr, source_ber, target_ber, target_ber - source_ber))
            logf.flush()



if __name__ == "__main__":
    # native_stft_eval(16)
    # for extf in [16]:
    #     blong_eval(extf)
    # blong_eval(16, 0)
    # blong_eval(16, 0)
    # eval_all(16)
    for extf in [32, 64, 128]:
        native_stft_eval(extf)

        






