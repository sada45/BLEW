import numpy as np
import torch 
from scripts.DANF.blong_nn_components import *
import scripts.raw_data_preprocessing as rdp
import config
from DANF.ble_data_loader import *
from torch.utils.data import DataLoader
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


best_model_list = ["blong_nn_extf=1_round=0_final.pt", "blong_nn_extf=2_round=0_final.pt", "blong_nn_extf=4_round=0_final.pt","blong_nn_extf=8_round=2_final_0.pt", "blong_nn_extf=16_round=4_final_0.pt", "blong_nn_extf=32_round=3_final.pt", "blong_nn_extf=64_round=3_final.pt", "blong_nn_extf=128_round=4_final.pt"]

batchs = []

def symbol_stft_decoding(spec):
    rem_num = spec.shape[2] // 2
    bit_data = torch.zeros(spec.shape[0], dtype=torch.uint8)
    spec = spec[:, 0, :, :] + spec[:, 1, :, :]
    high_freq = torch.sum(spec[:, :rem_num, :], dim=(1, 2))
    low_freq = torch.sum(spec[:, rem_num: 2*rem_num], dim=(1, 2))
    freq_diff = high_freq - low_freq
    bit_1_idx = torch.where(freq_diff > 0)[0]
    bit_data[bit_1_idx] = 1
    return bit_data

def danf_runtime_test(extf, batch_size=64, native=False, test_num=100):
    global batchs
    model_name = best_model_list[int(np.log2(extf))]
    model_states = torch.load("./models/models/" + model_name)
    fe = FeatureExtractor().to(device)
    nf = NoiseFilter(extf).to(device)
    fe.load_state_dict(model_states["fe_state"])
    nf.load_state_dict(model_states["nf_state"])
    fe.eval()
    nf.eval()
    # dataset = ble_raw_dataset(extf, True, 0)
    # data_loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=8)
    # dataset_iter = iter(data_loader)
    # batch, label = next(dataset_iter)
    # batch = batch.to(device)
    batch = batchs[int(np.log2(extf))].to(device)
    with torch.no_grad():
        st = time.time()
        for _ in range(test_num):
            stft_signals = rdp.get_stft_symbols(batch, extf)
            stft_signals = torch.abs(stft_signals)
            stft_signals = rdp.stft_normalize(stft_signals)
            if native:
                symbol_stft_decoding(stft_signals)
            else:
                features = fe(stft_signals)
                denoise_spec = nf(features) * stft_signals
                symbol_stft_decoding(denoise_spec)
        et = time.time()
        return (et - st) / test_num / batch_size  # return the time or each symbol

def test_init(batch_size=64):
    global batchs
    for extf in [1, 2, 4, 8, 16, 32, 64, 128]:
        dataset = ble_raw_dataset(extf, True, 0)
        data_loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=8)
        dataset_iter = iter(data_loader)
        batch, label = next(dataset_iter)
        batchs.append(batch)
        del dataset_iter, data_loader, dataset
        

if __name__== "__main__":
    test_init()
    # for extf in [1, 2, 4, 8, 16, 32, 64, 128]:
    #     runtime = danf_runtime_test(extf, native=False)
    #     print(extf, runtime)
    # print("=============================")
    # for extf in [4, 2, 1, 8, 16, 32, 64, 128]:
    #     runtime = danf_runtime_test(extf, native=False, test_num=1000)
    #     print(extf, runtime)
    for extf in [1,2, 4, 8, 16, 32, 64, 128]:
        runtime = danf_runtime_test(extf, native=True, test_num=100)
    print("Native=======================")
    for extf in [128, 64, 32, 16, 8, 4, 2, 1]:
        runtime = danf_runtime_test(extf, native=True, test_num=1000)
        print(extf, runtime)






