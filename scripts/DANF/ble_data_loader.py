import numpy as np
import torch
import sys
sys.path.append("./scripts/")
import config
import raw_data_preprocessing as raw_data_processing
import os
from torch.utils.data import Dataset
import utils

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
This code is used to dynamically load signals from disk and
preprocess them into STFT results
"""
class ble_torch_dataset(Dataset):
    def __init__(self, extf, train, snr_range):
        file_names = os.listdir("./processed_data/extf=" + str(extf))
        self.extf = extf
        self.snr_range = snr_range
        self.train = train
        self.sym_num = 0
        for file_name in file_names:
            if train and file_name.find("test") == -1:
                self.sym_num += config.file_batch_size
            elif not train and file_name.find("test") != -1:
                self.sym_num += config.file_batch_size
        if not isinstance(snr_range, list):
            self.fix_snr = True
            self.db = torch.zeros((1,1), dtype=torch.float32)
            self.db[0, 0] = self.snr_range
        else:
            self.fix_snr = False
    
    def set_snr(self, snr):
        self.snr_range = snr
        if isinstance(snr, list): 
            self.fix_snr = False
        else:
            self.fix_snr = True

    def __len__(self):
        return self.sym_num
    
    def __getitem__(self, index):
        if self.train:
            data = torch.load("./processed_data/extf=" + str(self.extf) + "/" + str(index) + ".pt")
        else:
            data = torch.load("./processed_data/extf=" + str(self.extf) + "/test_" + str(index) + ".pt")
        if self.fix_snr:
            db = self.db
        else:
            db = torch.randint(self.snr_range[0], self.snr_range[1], (1, 1))
        signal_with_noise = raw_data_processing.symbol_add_noise(data["raw"], data["noise"], db)
        noise_symbol_stft = raw_data_processing.get_stft_symbols(signal_with_noise, self.extf)
        # The abs() put here will not change the amplitude of IQ signal
        # If not, the noise level will be lifted
        noise_symbol_stft = torch.abs(noise_symbol_stft)
        noise_symbol_stft = raw_data_processing.stft_normalize(noise_symbol_stft)
        symbol_stft = raw_data_processing.get_stft_symbols( data["raw"], self.extf)
        symbol_stft = torch.abs(symbol_stft)
        symbol_stft = raw_data_processing.stft_normalize(symbol_stft)
        chan_label = torch.zeros(40, dtype=torch.float32)
        chan_label[int(data["chan"][0])] = 1.0
        # if self.train:
        #     return noise_symbol_stft.squeeze(0), symbol_stft.squeeze(0), data["label"].squeeze(0), chan_label, db.squeeze(0)
        # else:
        return noise_symbol_stft.squeeze(0), symbol_stft.squeeze(0), data["label"].squeeze(0), data["chan"].squeeze(0), db.squeeze(0)


class ble_raw_dataset(Dataset):
    def __init__(self, f_name, extf, snr_range, fast=False):
        self.extf = extf
        self.snr_range = snr_range
        self.sym_num = 0
        if snr_range is not None and not isinstance(snr_range, list):
            self.fix_snr = True
        else:
            self.fix_snr = False
        data = np.load("./processed_data/" + f_name + ".npz")
        self.sym = data["f_sym"]
        self.label = data["label"]
        self.noise = data["noise"]
        if fast:
            self.sym = self.sym[:1000]
            self.label = self.label[:1000]
            self.noise = self.noise[:1000]

    def set_snr(self, snr):
        self.snr_range = snr
        if isinstance(snr, list): 
            self.fix_snr = False
        else:
            self.fix_snr = True
    
    def set_len(self, l):
        if l > self.sym.shape[0]:
            raise Exception("No enough data")
        elif l == self.sym.shape[0]:
            return 
        else:
            self.sym = self.sym[:l]
            self.label = self.label[:l]
            self.noise = self.noise[:l]

    def __len__(self):
        return int(self.sym.shape[0])
    
    def __getitem__(self, index):
        if self.fix_snr:
            db = self.snr_range
        else:
            db = np.random.randint(self.snr_range[0], self.snr_range[1], 1)
        # bak = self.noise[index:index+1, :].copy()
        # print("before", (bak==self.noise[index:index+1, :]).all())
        signal_with_noise = utils.awgn(self.sym[index:index+1, :], db, noise_signal=self.noise[index:index+1, :])
        # print("after", (bak==self.noise[index:index+1, :]).all())
        # signal_with_noise = signal_with_noise[0, :]
        signal_with_noise = utils.signal_normalize(signal_with_noise, 1)[0, :].astype(np.complex64)
        filtered_signal = utils.filter(signal_with_noise).astype(np.complex64)
        signal_with_noise = signal_with_noise[config.extra:len(signal_with_noise)-config.extra]
        filtered_signal = filtered_signal[config.extra:len(filtered_signal)-config.extra]
        raw_array = np.vstack([signal_with_noise.real, signal_with_noise.imag, filtered_signal.real, filtered_signal.imag])
        # one_hot_label = torch.zeros(2, dtype=torch.float32)
        # one_hot_label[int(self.label[index])] = 1
        return torch.from_numpy(filtered_signal.copy()), torch.from_numpy(raw_array), torch.from_numpy(self.label[index:index+1])

class ble_true_dataset(Dataset):
    def __init__(self, f_name, extf, fast=False):
        self.extf = extf
        self.sym_num = 0
        data = np.load("./processed_data/" + f_name + ".npz")
        self.sym = data["sym"]
        self.label = data["label"].astype(np.float32)
        self.snr = data["snr"]
        if fast:
            idx = np.random.permutation(self.sym.shape[0])[:5000]
            self.sym = self.sym[idx]
            self.label = self.label[idx]
            self.snr = self.snr[idx]
        self.avi_idx = np.arange(self.sym.shape[0])

    def set_len(self, l):
        if l > self.sym.shape[0]:
            raise Exception("No enough data")
        elif l == self.sym.shape[0]:
            return 
        else:
            self.sym = self.sym[:l]
            self.label = self.label[:l]
    
    def set_snr(self, target_snr):
        if target_snr is None:
            self.avi_idx = np.arange(self.sym.shape[0])
        elif isinstance(target_snr, list):
            self.avi_idx = np.where((self.snr>=target_snr[0]) & (self.snr<=target_snr[1]))[0]
        else:
            self.avi_idx = np.where(self.snr==target_snr)[0]

    def __len__(self):
        return len(self.avi_idx)
    
    def __getitem__(self, index):
        i = self.avi_idx[index]
        sym = utils.signal_normalize(self.sym[i:i+1, :], 1)[0, :].astype(np.complex64)
        f_sym = utils.filter(sym).astype(np.complex64)
        sym = sym[config.extra:len(sym)-config.extra]
        f_sym = f_sym[config.extra:len(f_sym)-config.extra]
        raw_array = np.vstack([sym.real, sym.imag, f_sym.real, f_sym.imag])
        return torch.from_numpy(f_sym.copy()), torch.from_numpy(raw_array), torch.from_numpy(self.label[i:i+1])


class ble_ram_dataset(Dataset):
    def __init__(self, syms, labels, extf, db_dis=None):
        self.extf = extf
        self.syms = syms
        self.labels = labels
        self.db_dis = db_dis
        self.avi_idx = np.arange(self.syms.shape[0])

    def __len__(self):
        return int(len(self.avi_idx))
    
    def set_snr(self, snr):
        if self.db_dis is not None:
            self.avi_idx = np.where(self.db_dis==snr)[0]

    def __getitem__(self, index):
        i = self.avi_idx[index]
        sym = utils.signal_normalize(self.syms[i:i+1, :], 1)[0, :].astype(np.complex64)
        f_sym = utils.filter(sym).astype(np.complex64)
        sym = sym[config.extra:len(sym)-config.extra]
        f_sym = f_sym[config.extra:len(f_sym)-config.extra]
        raw_array = np.vstack([sym.real, sym.imag, f_sym.real, f_sym.imag])
        if self.labels is not None:
            return torch.from_numpy(f_sym.copy()), torch.from_numpy(raw_array), torch.from_numpy(self.labels[i:i+1])
        else:
            return torch.from_numpy(f_sym.copy()), torch.from_numpy(raw_array)










