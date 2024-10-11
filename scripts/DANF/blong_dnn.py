import numpy as np
import torch
from DANF.blong_nn_components import *
import DANF.ble_data_loader as ble_data_loader
from torch.utils.data import DataLoader
import raw_data_preprocessing as rdp
import utils

# best_steps = {1:5000, 2:5600, 4:4900, 8:6200, 16:5000, 32:5200, 64:3600}
best_steps = {1:5000, 2: 5600, 4: 3600, 8: 4000, 16: 4000, 32: 4200, 64:3400}
f = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# To make sure the model reproducibility
SEED = 11
np.random.seed(SEED)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-10.*p)) - 1.

def naive_decoder_eval(extf, test_dataset, db_range=None):
    global f
    if db_range is None:
        if extf <= 4:
            db_range = np.arange(-15, 1, dtype=np.int32)
        else:
            db_range = np.arange(-20, -9, dtype=np.int32)
    for db in db_range:
        test_dataset.set_snr(db)
        test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=8)
        correct_num = np.zeros(2)
        total_num = np.zeros(2)
        for idx, (noise_signal, raw_chan, label) in enumerate(test_data_loader):
            label = label.squeeze(1)
            stft_res = rdp.get_stft_symbols(noise_signal, extf, cpu=True)
            stft_bits = utils.stft_decode(stft_res)
            pf_bits = utils.phase_diff_decode(raw_chan[:, 2:, :])
            correct_num[0] += len(torch.where(stft_bits==label)[0])
            correct_num[1] += len(torch.where(pf_bits==label)[0])
            total_num += noise_signal.shape[0]
        ber = (total_num - correct_num) / total_num
        s = "snr={}dB, stft_ber={}, phase_ber={}".format(db, ber[0], ber[1])
        print(s)
        if f is not None:
            f.write(s + "\n")

def nn_decoder_eval(sfe, rfe, lfe, bd, extf, test_dataset, db_range=None):
    global f
    sfe.eval()
    rfe.eval()
    lfe.eval()
    bd.eval()
    if db_range is None:
        if extf <= 4:
            db_range = np.arange(-15, 1, dtype=np.int32)
        else:
            db_range = np.arange(-20, -9, dtype=np.int32)
    with torch.no_grad():
        for db in db_range:
            test_dataset.set_snr(db)
            test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
            correct_num = 0
            total_num = 0
            ber0_counter = 0
            for idx, (noise_signal, raw_chan, label) in enumerate(test_data_loader):
                stft_res = rdp.get_stft_symbols(noise_signal, extf, cpu=False)
                raw_chan = raw_chan.to(device)
                stft_feature = sfe(stft_res)
                raw_feature = rfe(raw_chan)
                lstm_feature = lfe(stft_feature, raw_feature)
                decode_poss = bd(lstm_feature)
                decode_poss = torch.sigmoid(decode_poss).cpu()
                decode_bits = torch.zeros(noise_signal.shape[0], dtype=torch.float32)
                decode_bits[torch.where(decode_poss>=0.5)[0]] = 1
                label = label.squeeze(1)
                correct_num += torch.where(decode_bits==label)[0].shape[0]
                total_num += noise_signal.shape[0]
            ber = (total_num - correct_num) / total_num
            s = "snr={}dB, BER={}".format(db, ber)
            print(s)
            if f is not None:
                f.write(s + "\n")
            if ber == 0:
                ber0_counter += 1
                if ber0_counter >= 3:
                    print(ber0_counter)
                    # Already get the optimal
                    break
    sfe.train()
    rfe.train()
    lfe.train()
    bd.train()

""""
init DNN
snr=-20dB, BER=0.362
snr=-19dB, BER=0.34
snr=-18dB, BER=0.31
snr=-17dB, BER=0.28
snr=-16dB, BER=0.249
snr=-15dB, BER=0.21
snr=-14dB, BER=0.159
snr=-13dB, BER=0.113
snr=-12dB, BER=0.081
snr=-11dB, BER=0.056
snr=-10dB, BER=0.033
"""
def danf_train(dnn_dict, source_domain_data_loader, target_domain_data_loader, test_dataset, extf, epoch=5, lamb=1, name="danf"):
    global f
    f = open("./models/train_logs/danf_{}e.log".format(extf), "w")
    sfe = STFTFeatureExtractor(extf).to(device)
    sfe.load_state_dict(dnn_dict["sfe"])
    rfe = RawFeatureExtractor(extf).to(device)
    rfe.load_state_dict(dnn_dict["rfe"])
    lfe = LSTMFeatureExtractor(extf).to(device)
    lfe.load_state_dict(dnn_dict["lfe"])
    bd = Discriminator(extf).to(device).to(device)
    bd.load_state_dict(dnn_dict["bd"])
    dd = Discriminator(extf).to(device)
    print("init DNN===============")
    f.write("init DNN===============")
    nn_decoder_eval(sfe, rfe, lfe, bd, extf, test_dataset)
    # print("naive==================")
    # f.write("naive==================")
    # naive_decoder_eval(extf, test_dataset)
    # return 

    optimizer_sfe = torch.optim.Adam(sfe.parameters(), 1e-4, [0.9, 0.999])
    optimizer_rfe = torch.optim.Adam(rfe.parameters(), 1e-4, [0.9, 0.999])
    optimizer_lfe = torch.optim.Adam(lfe.parameters(), 1e-4, [0.9, 0.999])
    optimizer_bd = torch.optim.Adam(bd.parameters(), 1e-4, [0.9, 0.999])
    optimizer_dd = torch.optim.Adam(dd.parameters(), 1e-4, [0.9, 0.999])
    dd_loss_spec = torch.nn.BCEWithLogitsLoss(reduction="mean")
    bd_loss_spec = torch.nn.BCEWithLogitsLoss(reduction="mean")

    step = 0
    for e in range(epoch):
        l = lamb * get_lambda(e+1, epoch)
        print(l)
        for idx, ((s_noise_signal, s_raw_chan, s_label), (t_noise_signal, t_raw_chan, t_label)) in enumerate(zip(source_domain_data_loader, target_domain_data_loader)):
            s_stft_res = rdp.get_stft_symbols(s_noise_signal, extf, cpu=False)
            t_stft_res = rdp.get_stft_symbols(t_noise_signal, extf, cpu=False)
            stft_res = torch.cat([s_stft_res, t_stft_res], dim=0)
            s_label = s_label.to(device)
            domain_label = torch.zeros([stft_res.shape[0], 1], dtype=torch.float32).to(device)
            domain_label[s_stft_res.shape[0]:, 0] = 1
            raw_chan = torch.cat([s_raw_chan, t_raw_chan], dim=0).to(device)
            # Calculate the stft result
            stft_feature = sfe(stft_res)
            raw_feature = rfe(raw_chan)
            lstm_feature = lfe(stft_feature, raw_feature)
            # Predict and optimzie the domain discriminator
            d = dd(lstm_feature.detach())
            dd_loss = dd_loss_spec(d, domain_label)
            dd.zero_grad()
            dd_loss.backward()
            optimizer_dd.step()
            # optimize the decoder
            b = bd(lstm_feature[:s_noise_signal.shape[0]])
            d = dd(lstm_feature)
            bd_loss = bd_loss_spec(b, s_label)
            dd_loss = dd_loss_spec(d, domain_label)
            total_loss = bd_loss - l * dd_loss
            sfe.zero_grad()
            rfe.zero_grad()
            lfe.zero_grad()
            bd.zero_grad()
            total_loss.backward()
            optimizer_sfe.step()
            optimizer_rfe.step()
            optimizer_lfe.step()
            optimizer_bd.step()
            step += 1
            if step % 200 == 0:
                print("===============")
                s = "dnn, epoch:{}, step:{}, loss:{:.4f}, dd_loss:{:.4f}".format(e, step, bd_loss.item(), dd_loss.item())
                print(s)
                if f is not None:
                    f.write("========================\n")
                    f.write(s + "\n")
                nn_decoder_eval(sfe, rfe, lfe, bd, extf, test_dataset)
                d = {"sfe": sfe.state_dict(), "rfe": rfe.state_dict(), "lfe": lfe.state_dict(), "bd": bd.state_dict()}
                torch.save(d, "./models/{}_{}e_{}s.pt".format(name, extf, step))

def dnn_train(source_domain_data_loader, test_dataset, extf, epoch=5, name="dnn"):
    global f
    f = open("./models/train_logs/{}_{}e.log".format(name, extf), "w")
    sfe = STFTFeatureExtractor(extf).to(device)
    rfe = RawFeatureExtractor(extf).to(device)
    lfe = LSTMFeatureExtractor(extf).to(device)
    bd = Discriminator(extf).to(device)
    optimizer_sfe = torch.optim.Adam(sfe.parameters(), 1e-4, [0.9, 0.999])
    optimizer_rfe = torch.optim.Adam(rfe.parameters(), 1e-4, [0.9, 0.999])
    optimizer_lfe = torch.optim.Adam(lfe.parameters(), 1e-4, [0.9, 0.999])
    optimizer_bd = torch.optim.Adam(bd.parameters(), 1e-4, [0.9, 0.999])
    bd_loss_spec = torch.nn.BCEWithLogitsLoss(reduction="mean")

    step = 0
    for e in range(epoch):
        for idx, (noise_signal, raw_chan, label) in enumerate(source_domain_data_loader):
            stft_res = rdp.get_stft_symbols(noise_signal, extf, cpu=False)
            raw_chan = raw_chan.to(device)
            label = label.to(device)
            stft_feature = sfe(stft_res)
            raw_feature = rfe(raw_chan)
            lstm_feature = lfe(stft_feature, raw_feature)
            decode = bd(lstm_feature)
            bd_loss = bd_loss_spec(decode, label)
            sfe.zero_grad()
            rfe.zero_grad()
            lfe.zero_grad()
            bd.zero_grad()
            bd_loss.backward()
            optimizer_bd.step()
            optimizer_lfe.step()
            optimizer_sfe.step()
            optimizer_rfe.step()
            step += 1
            if step % 200 == 0:
                s = "dnn, epoch:{}, step:{}, loss:{:.4f}".format(e, step, bd_loss.item())
                print("====================")
                print(s)
                if f is not None:
                    f.write("====================\n")
                    f.write(s + "\n")
                if step > 400:
                    nn_decoder_eval(sfe, rfe, lfe, bd, extf, test_dataset)
                    d = {"sfe": sfe.state_dict(), "rfe": rfe.state_dict(), "lfe": lfe.state_dict(), "bd": bd.state_dict()}
                    torch.save(d, "./models/{}_{}e_{}s.pt".format(name, extf, step))
batch_size = 64
best_dnn_round = {1:5000, 2: 6800, 4: 7000, 8: 6800, 16:6400, 32:6400, 64:3400}
def danf_train_main(extf, synthesis):
    dataset_prefix = "white_noise_{}e_".format(extf)
    target_prefix = "wifi_int_{}e_".format(extf)
    save_name = "danf"
    source_dataset = ble_data_loader.ble_raw_dataset(dataset_prefix + "train", extf, [-20, 0])
    if synthesis:
        target_dataset = ble_data_loader.ble_raw_dataset(target_prefix + "train", extf, [-20, 0])
        test_dataset = ble_data_loader.ble_raw_dataset(target_prefix + "test", extf, -20, fast=True)
    else:
        target_dataset = ble_data_loader.ble_true_dataset(target_prefix, extf)
        test_dataset = ble_data_loader.ble_true_dataset(target_prefix, extf, fast=True)
    cut_len = np.min([len(source_dataset), len(target_dataset)])
    source_dataset.set_len(cut_len)
    target_dataset.set_len(cut_len)

    dnn_dict = torch.load("./models/dnn_{}e_{}s.pt".format(extf, best_dnn_round[extf]))
    source_dataloader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    danf_train(dnn_dict, source_dataloader, target_dataloader, test_dataset, extf, name=save_name)
    if f is not None:
        f.flush()
        f.close()

def dnn_train_main(extf):
    dataset_prefix = "white_noise_{}e_".format(extf)
    train_dataset = ble_data_loader.ble_raw_dataset(dataset_prefix + "train", extf, [-20, 0])
    test_dataset = ble_data_loader.ble_raw_dataset(dataset_prefix + "test", extf, -10, True)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    # test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
    dnn_train(train_data_loader, test_dataset, extf)
    if f is not None:
        f.flush()
        f.close()
    
# The input is [packet num, bits num, signal len]
# Decode a packet
def dnn_decode(raw_signal, filtered_signal, extf, data):
    data = torch.from_numpy(data)
    if isinstance(raw_signal, np.ndarray):
        raw_signal = torch.from_numpy(raw_signal)
    if isinstance(filtered_signal, np.ndarray):
        filtered_signal = torch.from_numpy(filtered_signal)
    raw_array = torch.zeros([filtered_signal.shape[0], filtered_signal.shape[1], 4, filtered_signal.shape[2]], dtype=torch.float32)
    raw_array[:, :, 0, :] = raw_signal[: ,:, :].real
    raw_array[:, :, 1, :] = raw_signal[: ,:, :].imag
    raw_array[:, :, 2, :] = filtered_signal[: ,:, :].real
    raw_array[:, :, 3, :] = filtered_signal[: ,:, :].imag

    # Load the DNN
    dnn_dict = torch.load("./models/dnn_{}e_{}s.pt".format(extf, best_dnn_round[extf]))
    sfe = STFTFeatureExtractor(extf).to(device)
    sfe.load_state_dict(dnn_dict["sfe"])
    rfe = RawFeatureExtractor(extf).to(device)
    rfe.load_state_dict(dnn_dict["rfe"])
    lfe = LSTMFeatureExtractor(extf).to(device)
    lfe.load_state_dict(dnn_dict["lfe"])
    bd = Discriminator(extf).to(device).to(device)
    bd.load_state_dict(dnn_dict["bd"])
    sfe.eval()
    rfe.eval()
    lfe.eval()
    bd.eval()

    # Start decoding
    correct_pkt_num = 0
    stft_cpn = 0
    pd_cpn = 0
    bcr = 0
    stft_bcr = 0
    pd_bcr = 0
    dnn_pbcr = []
    stft_pbcr = []
    pd_pbcr = []
    with torch.no_grad():
        for i in range(raw_signal.shape[0]):
            stft_res = rdp.get_stft_symbols(filtered_signal[i, :, :], extf, cpu=False)
            raw_chan = raw_array[i, :, :, :].to(device)
            stft_feature = sfe(stft_res)
            raw_feature = rfe(raw_chan)
            lstm_feature = lfe(stft_feature, raw_feature)
            dnn_poss = bd(lstm_feature).squeeze(1)
            dnn_poss = torch.sigmoid(dnn_poss)
            dnn_decode = torch.zeros(raw_signal.shape[1], dtype=torch.uint8)
            dnn_decode[torch.where(dnn_poss>=0.5)] = 1
            # print(utils.bits_to_bytes(dnn_decode))
            dnn_corr = torch.eq(dnn_decode, data)
            if dnn_corr.all():
                correct_pkt_num += 1
            dnn_pbcr.append(torch.sum(dnn_corr))  
            bcr += dnn_pbcr[-1]
            stft_decode = utils.stft_decode(stft_res)
            # print(utils.bits_to_bytes(stft_decode))
            pd_decode = utils.phase_diff_decode(raw_chan)
            stft_corr = torch.eq(stft_decode, data)
            pd_corr = torch.eq(pd_decode, data)
            # print(torch.where(stft_corr==False))
            if stft_corr.all():
                stft_cpn += 1
            stft_pbcr.append(torch.sum(stft_corr))
            stft_bcr += stft_pbcr[-1]
            if pd_corr.all():
                pd_cpn += 1
            pd_pbcr.append(torch.sum(pd_corr))
            pd_bcr += pd_pbcr[-1]
    return np.array([correct_pkt_num, stft_cpn, pd_cpn]), np.array([bcr, stft_bcr, pd_bcr]), np.array([dnn_pbcr, stft_pbcr, pd_pbcr])

def decode_ber(syms, labels, extf, model_name=None):
    dataset = ble_data_loader.ble_ram_dataset(syms, labels, extf)
    dataloader = DataLoader(dataset, batch_size=64, drop_last=False, shuffle=False, num_workers=8)
    if model_name is None: 
        dnn_dict = torch.load("./models/dnn_{}e_{}s.pt".format(extf, best_dnn_round[extf]))
    else:
        dnn_dict = torch.load("./models/" + model_name)
    sfe = STFTFeatureExtractor(extf).to(device)
    sfe.load_state_dict(dnn_dict["sfe"])
    rfe = RawFeatureExtractor(extf).to(device)
    rfe.load_state_dict(dnn_dict["rfe"])
    lfe = LSTMFeatureExtractor(extf).to(device)
    lfe.load_state_dict(dnn_dict["lfe"])
    bd = Discriminator(extf).to(device).to(device)
    bd.load_state_dict(dnn_dict["bd"])
    sfe.eval()
    rfe.eval()
    lfe.eval()
    bd.eval()
    
    dnn_bits = []
    stft_bits = []
    pd_bits = []

    with torch.no_grad():
        for _, (f_sym, raw_chan, _) in enumerate(dataloader):
            stft_res = rdp.get_stft_symbols(f_sym, extf, cpu=False)
            raw_chan = raw_chan.to(device)
            stft_feature = sfe(stft_res)
            raw_feature = rfe(raw_chan)
            lstm_feature = lfe(stft_feature, raw_feature)
            dnn_poss = bd(lstm_feature).squeeze(1)
            dnn_poss = torch.sigmoid(dnn_poss)
            dnn_decode = torch.zeros(f_sym.shape[0], dtype=torch.uint8)
            dnn_decode[torch.where(dnn_poss>=0.5)] = 1
            dnn_bits.append(dnn_decode)
            stft_decode = utils.stft_decode(stft_res)
            pd_decode = utils.phase_diff_decode(raw_chan)
            stft_bits.append(stft_decode)
            pd_bits.append(pd_decode)
    dnn_bits = torch.cat(dnn_bits).numpy()
    stft_bits = torch.cat(stft_bits).numpy()
    pd_bits = torch.cat(pd_bits).numpy()
    correct_num = np.zeros(3)
    correct_num[0] += np.sum(np.equal(dnn_bits, labels))
    correct_num[1] += np.sum(np.equal(stft_bits, labels))
    correct_num[2] += np.sum(np.equal(pd_bits, labels))
    return [dnn_bits, stft_bits, pd_bits], correct_num

def stft_decode_bits(syms, extf, data_bits):
    # best_steps = {2: 5600, 4: 3600, 8: 4000, 16: 4000, 32: 4200, 64:3400}
    data_bits = torch.from_numpy(data_bits)
    dataset = ble_data_loader.ble_ram_dataset(syms, None, extf)
    dataloader = DataLoader(dataset, batch_size=len(data_bits), drop_last=False, shuffle=False, num_workers=8)
    
    stft_valid_idx = []
    total_num = 0
    corr_num = 0
    counter = 0
    for _, (f_sym, raw_chan) in enumerate(dataloader):
        stft_res = rdp.get_stft_symbols(f_sym, extf, cpu=False)
        stft_decode = utils.stft_decode(stft_res)
        if torch.eq(stft_decode, data_bits).all():
            stft_valid_idx.append(counter)
        total_num += len(data_bits)
        corr_num += torch.sum(torch.eq(stft_decode, data_bits))
        counter += 1
    return np.array(stft_valid_idx), [corr_num, total_num]

def native_decode_bits(syms, extf, data_bits):
    # best_steps = {2: 5600, 4: 3600, 8: 4000, 16: 4000, 32: 4200, 64:3400}
    data_bits = torch.from_numpy(data_bits)
    dataset = ble_data_loader.ble_ram_dataset(syms, None, extf)
    dataloader = DataLoader(dataset, batch_size=len(data_bits), drop_last=False, shuffle=False, num_workers=8)
    
    native_valid_idx = []
    stft_valid_idx = []
    
    counter = 0
    all_bits_num = 0
    corr_bits_num = 0
    for i, (f_sym, raw_chan) in enumerate(dataloader):
        stft_res = rdp.get_stft_symbols(f_sym, extf, cpu=False)
        stft_decode = utils.stft_decode(stft_res)
        pd_decode = utils.phase_diff_decode(raw_chan)
        corr_bits_num += torch.sum(torch.eq(pd_decode, data_bits))
        all_bits_num += len(data_bits)
        if torch.eq(pd_decode, data_bits).all():
            native_valid_idx.append(counter)
        if torch.eq(stft_decode, data_bits).all():
            stft_valid_idx = []
        counter += 1
    # print(corr_bits_num / all_bits_num)
    return np.array(stft_valid_idx), np.array(native_valid_idx), [corr_bits_num, all_bits_num]

def dnn_decode_bits(syms, extf, data_bits, getdet=False, model_name=None):
    # best_steps = {2: 5600, 4: 3600, 8: 4000, 16: 4000, 32: 4200, 64:3400}
    data_bits = torch.from_numpy(data_bits)
    dataset = ble_data_loader.ble_ram_dataset(syms, None, extf)
    dataloader = DataLoader(dataset, batch_size=len(data_bits), drop_last=False, shuffle=False, num_workers=8)
    if model_name is None:
        dnn_dict = torch.load("/liymdata/liym/BLong/models/dnn_wr_{}e_{}s.pt".format(extf, best_steps[extf]))
    else:
        dnn_dict = torch.load("/liymdata/liym/BLong/models/" + model_name)
    sfe = STFTFeatureExtractor(extf).to(device)
    sfe.load_state_dict(dnn_dict["sfe"])
    rfe = RawFeatureExtractor(extf).to(device)
    rfe.load_state_dict(dnn_dict["rfe"])
    lfe = LSTMFeatureExtractor(extf).to(device)
    lfe.load_state_dict(dnn_dict["lfe"])
    bd = Discriminator(extf).to(device).to(device)
    bd.load_state_dict(dnn_dict["bd"])
    sfe.eval()
    rfe.eval()
    lfe.eval()
    bd.eval()
    
    valid_idx = []
    stft_valid_idx = []
    counter = 0
    with torch.no_grad():
        for _, (f_sym, raw_chan) in enumerate(dataloader):
            stft_res = rdp.get_stft_symbols(f_sym, extf, cpu=False)
            raw_chan = raw_chan.to(device)
            stft_feature = sfe(stft_res)
            raw_feature = rfe(raw_chan)
            lstm_feature = lfe(stft_feature, raw_feature)
            dnn_poss = bd(lstm_feature).squeeze(1)
            dnn_poss = torch.sigmoid(dnn_poss)
            dnn_decode = torch.zeros(f_sym.shape[0], dtype=torch.uint8)
            dnn_decode[torch.where(dnn_poss>=0.5)] = 1
            stft_decode = utils.stft_decode(stft_res)
            if torch.eq(dnn_decode, data_bits).all():
                valid_idx.append(counter)
            if torch.eq(stft_decode, data_bits).all():
                stft_valid_idx.append(counter)
            counter += 1
    print(len(valid_idx), len(stft_valid_idx))
    if getdet:
        return np.array(valid_idx), np.array(stft_valid_idx), [len(valid_idx), len(stft_valid_idx)]
    else:
        return np.array(valid_idx)

def dnn_decode_throghput(syms, extf, data_bits, model_name=None):
    # best_steps = {2: 5600, 4: 3600, 8: 4000, 16: 4000, 32: 4200, 64:3400}
    data_bits = torch.from_numpy(data_bits)
    dataset = ble_data_loader.ble_ram_dataset(syms, None, extf)
    dataloader = DataLoader(dataset, batch_size=len(data_bits), drop_last=False, shuffle=False, num_workers=8)
    if model_name is None:
        dnn_dict = torch.load("./models/dnn_wr_{}e_{}s.pt".format(extf, best_steps[extf]))
    else:
        dnn_dict = torch.load("./models/" + model_name)
        
    sfe = STFTFeatureExtractor(extf).to(device)
    sfe.load_state_dict(dnn_dict["sfe"])
    rfe = RawFeatureExtractor(extf).to(device)
    rfe.load_state_dict(dnn_dict["rfe"])
    lfe = LSTMFeatureExtractor(extf).to(device)
    lfe.load_state_dict(dnn_dict["lfe"])
    bd = Discriminator(extf).to(device).to(device)
    bd.load_state_dict(dnn_dict["bd"])
    sfe.eval()
    rfe.eval()
    lfe.eval()
    bd.eval()
    
    valid_idx = []
    stft_valid_idx = []
    counter = 0
    total_num = 0
    stft_corr_num = 0
    dnn_corr_num = 0
    with torch.no_grad():
        for _, (f_sym, raw_chan) in enumerate(dataloader):
            stft_res = rdp.get_stft_symbols(f_sym, extf, cpu=False)
            raw_chan = raw_chan.to(device)
            stft_feature = sfe(stft_res)
            raw_feature = rfe(raw_chan)
            lstm_feature = lfe(stft_feature, raw_feature)
            dnn_poss = bd(lstm_feature).squeeze(1)
            dnn_poss = torch.sigmoid(dnn_poss)
            dnn_decode = torch.zeros(f_sym.shape[0], dtype=torch.uint8)
            dnn_decode[torch.where(dnn_poss>=0.5)] = 1
            stft_decode = utils.stft_decode(stft_res)
            if torch.eq(dnn_decode, data_bits).all():
                valid_idx.append(counter)
            if torch.eq(stft_decode, data_bits).all():
                stft_valid_idx.append(counter)
            total_num += len(data_bits)
            stft_corr_num += torch.sum(torch.eq(stft_decode, data_bits))
            dnn_corr_num += torch.sum(torch.eq(dnn_decode, data_bits))
            counter += 1
    return np.array(valid_idx), np.array(stft_valid_idx), [dnn_corr_num, total_num], [stft_corr_num, total_num]
     
def dnn_preamble_detection_test(syms, extf, data_bits, model_name=None):
    # best_steps = {2: 5600, 4: 3600, 8: 4000, 16: 4000, 32: 4200, 64:3400}
    data_bits = torch.from_numpy(data_bits)
    dataset = ble_data_loader.ble_ram_dataset(syms, None, extf)
    dataloader = DataLoader(dataset, batch_size=len(data_bits), drop_last=False, shuffle=False, num_workers=8)
    if model_name is None:
        dnn_dict = torch.load("./models/dnn_wr_{}e_{}s.pt".format(extf, best_steps[extf]))
    else:
        dnn_dict = torch.load("./models/" + model_name)
        
    sfe = STFTFeatureExtractor(extf).to(device)
    sfe.load_state_dict(dnn_dict["sfe"])
    rfe = RawFeatureExtractor(extf).to(device)
    rfe.load_state_dict(dnn_dict["rfe"])
    lfe = LSTMFeatureExtractor(extf).to(device)
    lfe.load_state_dict(dnn_dict["lfe"])
    bd = Discriminator(extf).to(device).to(device)
    bd.load_state_dict(dnn_dict["bd"])
    sfe.eval()
    rfe.eval()
    lfe.eval()
    bd.eval()
    
    valid_idx = []
    stft_valid_idx = []
    pd_valid_idx = []
    counter = 0
    with torch.no_grad():
        for _, (f_sym, raw_chan) in enumerate(dataloader):
            stft_res = rdp.get_stft_symbols(f_sym, extf, cpu=False)
            raw_chan = raw_chan.to(device)
            stft_feature = sfe(stft_res)
            raw_feature = rfe(raw_chan)
            lstm_feature = lfe(stft_feature, raw_feature)
            dnn_poss = bd(lstm_feature).squeeze(1)
            dnn_poss = torch.sigmoid(dnn_poss)
            dnn_decode = torch.zeros(f_sym.shape[0], dtype=torch.uint8)
            dnn_decode[torch.where(dnn_poss>=0.5)] = 1
            stft_decode = utils.stft_decode(stft_res)
            pd_decode = utils.phase_diff_decode(raw_chan)
            # print(dnn_decode.reshape([-1, 8]))
            # print(stft_decode.reshape([-1, 8]))
            # print("===============")
            if torch.eq(dnn_decode, data_bits).all():
                valid_idx.append(counter)
            if torch.eq(stft_decode, data_bits).all():
                stft_valid_idx.append(counter)
            if torch.eq(pd_decode, data_bits).all():
                pd_valid_idx.append(counter)
            counter += 1
    return np.array([len(valid_idx), len(stft_valid_idx), len(pd_valid_idx)])



def eval_all(file_name, model_name, extf, snr_range):
    dnn_dict = torch.load("./models/" + model_name)
    sfe = STFTFeatureExtractor(extf).to(device)
    sfe.load_state_dict(dnn_dict["sfe"])
    rfe = RawFeatureExtractor(extf).to(device)
    rfe.load_state_dict(dnn_dict["rfe"])
    lfe = LSTMFeatureExtractor(extf).to(device)
    lfe.load_state_dict(dnn_dict["lfe"])
    bd = Discriminator(extf).to(device).to(device)
    bd.load_state_dict(dnn_dict["bd"])
    sfe.eval()
    rfe.eval()
    lfe.eval()
    bd.eval()

    dataset = ble_data_loader.ble_raw_dataset(file_name, extf, snr_range[0], fast=False)
    with torch.no_grad():
        bcrs = []
        for snr in snr_range:
            bcr = 0
            stft_bcr = 0
            pd_bcr = 0
            total_num = 0
            dataset.set_snr(snr)
            dataloader = DataLoader(dataset, batch_size=64, drop_last=False, shuffle=False, num_workers=8)
            for _, (f_sym, raw_chan, label) in enumerate(dataloader):
                stft_res = rdp.get_stft_symbols(f_sym, extf, cpu=False)
                raw_chan = raw_chan.to(device)
                stft_feature = sfe(stft_res)
                raw_feature = rfe(raw_chan)
                lstm_feature = lfe(stft_feature, raw_feature)
                dnn_poss = bd(lstm_feature).squeeze(1)
                dnn_poss = torch.sigmoid(dnn_poss)
                dnn_decode = torch.zeros(f_sym.shape[0], dtype=torch.uint8)
                dnn_decode[torch.where(dnn_poss>=0.5)] = 1
                label = label.squeeze(1)
                dnn_corr = torch.eq(dnn_decode, label)
                bcr += torch.sum(dnn_corr)
                stft_decode = utils.stft_decode(stft_res)
                pd_decode = utils.phase_diff_decode(raw_chan)
                stft_corr = torch.eq(stft_decode, label)
                pd_corr = torch.eq(pd_decode, label)
                stft_bcr += torch.sum(stft_corr)
                pd_bcr += torch.sum(pd_corr)
                total_num += f_sym.shape[0]
            bcrs.append(np.array([bcr, stft_bcr, pd_bcr]) / total_num)
    return np.array(bcrs)

def eval_goodput_snr(file_name, extf, snr, native):
    dnn_dict = torch.load("./models/dnn_{}e_{}s.pt".format(extf, best_dnn_round[extf]))
    sfe = STFTFeatureExtractor(extf).to(device)
    sfe.load_state_dict(dnn_dict["sfe"])
    rfe = RawFeatureExtractor(extf).to(device)
    rfe.load_state_dict(dnn_dict["rfe"])
    lfe = LSTMFeatureExtractor(extf).to(device)
    lfe.load_state_dict(dnn_dict["lfe"])
    bd = Discriminator(extf).to(device).to(device)
    bd.load_state_dict(dnn_dict["bd"])
    sfe.eval()
    rfe.eval()
    lfe.eval()
    bd.eval()

    dataset = ble_data_loader.ble_raw_dataset(file_name, extf, snr, fast=False)
    dataloader = DataLoader(dataset, batch_size=64, drop_last=False, shuffle=False, num_workers=8)
    dnn_bits = []
    stft_bits = []
    pd_bits = []
    labels = []
    with torch.no_grad():
        for _, (f_sym, raw_chan, label) in enumerate(dataloader):
            if native == 0:
                stft_res = rdp.get_stft_symbols(f_sym, extf, cpu=False)
                raw_chan = raw_chan.to(device)
                stft_feature = sfe(stft_res)
                raw_feature = rfe(raw_chan)
                lstm_feature = lfe(stft_feature, raw_feature)
                dnn_poss = bd(lstm_feature).squeeze(1)
                dnn_poss = torch.sigmoid(dnn_poss)
                dnn_decode = torch.zeros(f_sym.shape[0], dtype=torch.uint8)
                dnn_decode[torch.where(dnn_poss>=0.5)] = 1
                dnn_bits.append(dnn_decode)
            elif native == 1:
                stft_res = rdp.get_stft_symbols(f_sym, extf, cpu=False)
                stft_decode = utils.stft_decode(stft_res)
                stft_bits.append(stft_decode)
            else:
                pd_decode = utils.phase_diff_decode(raw_chan)
                pd_bits.append(pd_decode)
            labels.append(label.squeeze(1))
    if len(dnn_bits) == 0:
        dnn_bits = None
    else:
        dnn_bits = torch.cat(dnn_bits).numpy()
    if len(stft_bits) == 0:
        stft_bits = None
    else:
        stft_bits = torch.cat(stft_bits).numpy()
    if len(pd_bits) == 0:
        pd_bits = None
    else:
        pd_bits = torch.cat(pd_bits).numpy()
    labels = torch.cat(labels).numpy()
    return dnn_bits, stft_bits, pd_bits, labels

def get_dnn(model_name, extf):
    dnn_dict = torch.load("./models/" + model_name)
    dnn_sfe = STFTFeatureExtractor(extf).to(device)
    dnn_sfe.load_state_dict(dnn_dict["sfe"])
    dnn_rfe = RawFeatureExtractor(extf).to(device)
    dnn_rfe.load_state_dict(dnn_dict["rfe"])
    dnn_lfe = LSTMFeatureExtractor(extf).to(device)
    dnn_lfe.load_state_dict(dnn_dict["lfe"])
    dnn_bd = Discriminator(extf).to(device).to(device)
    dnn_bd.load_state_dict(dnn_dict["bd"])
    dnn_sfe.eval()
    dnn_rfe.eval()
    dnn_lfe.eval()
    dnn_bd.eval()
    dnn = [dnn_sfe, dnn_rfe, dnn_lfe, dnn_bd]
    return dnn

def dnn_infer(dnn, stft_res, raw_chan):
    stft_feature = dnn[0](stft_res)
    raw_feature = dnn[1](raw_chan)
    lstm_feature = dnn[2](stft_feature, raw_feature)
    dnn_poss = dnn[3](lstm_feature).squeeze(1)
    dnn_poss = torch.sigmoid(dnn_poss)
    dnn_decode = torch.zeros(stft_res.shape[0], dtype=torch.uint8)
    dnn_decode[torch.where(dnn_poss>=0.5)] = 1
    return dnn_decode

def eval_wifi(file_name, dnn_name, dnn_upd_name, extf, snr_range):
    dnn = get_dnn(dnn_name, extf)
    dnn_upd = get_dnn(dnn_upd_name, extf)

    dataset = ble_data_loader.ble_true_dataset(file_name, extf, fast=False)
    bcrs = []
    with torch.no_grad():
        for snr in snr_range:
            dnn_bcr = 0
            dnn_upd_bcr = 0
            stft_bcr = 0
            pd_bcr = 0
            total_num = 0
            dataset.set_snr(snr)
            dataloader = DataLoader(dataset, batch_size=64, drop_last=False, shuffle=False, num_workers=8)
            for _, (f_sym, raw_chan, label) in enumerate(dataloader):
                stft_res = rdp.get_stft_symbols(f_sym, extf, cpu=False)
                raw_chan = raw_chan.to(device)
                label = label.squeeze(1)
                dnn_decode = dnn_infer(dnn, stft_res, raw_chan)
                dnn_corr = torch.eq(dnn_decode, label)
                dnn_bcr += torch.sum(dnn_corr)
                dnn_upd_decode = dnn_infer(dnn_upd, stft_res, raw_chan)
                dnn_upd_corr = torch.eq(dnn_upd_decode, label)
                dnn_upd_bcr += torch.sum(dnn_upd_corr)
                stft_decode = utils.stft_decode(stft_res)
                pd_decode = utils.phase_diff_decode(raw_chan)
                stft_corr = torch.eq(stft_decode, label)
                pd_corr = torch.eq(pd_decode, label)
                stft_bcr += torch.sum(stft_corr)
                pd_bcr += torch.sum(pd_corr)
                total_num += f_sym.shape[0]
            bcrs.append(np.array([dnn_bcr, dnn_upd_bcr, stft_bcr, pd_bcr]) / total_num)
    return np.array(bcrs)



if __name__ == "__main__":
    db_ranges = []
    for extf in [1, 2, 4, 8, 16, 32, 64]:
        print("+++++++++++++++++++++++++++++++")
        print(extf)
        print("+++++++++++++++++++++++++++++++")
        danf_train_main(extf, synthesis=True)
