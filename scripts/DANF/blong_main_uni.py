import numpy as np
import torch
from blong_nn_components import *
import ble_data_loader as ble_data_loader
from torch.utils.data import DataLoader
import raw_data_preprocessing as rdp
import utils

f = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
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
        bers = []
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
            bers.append(ber)
            s = "snr={}dB, BER={}".format(db, ber)
            print(s)
            print(correct_num / total_num)
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
    return np.array(bers)

def danf_train(dnn_dict, source_domain_data_loader, target_domain_data_loader, test_dataset, extf, epoch=5, lamb=0.05, name="danf"):
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

def dnn_update(dnn_dict, source_domain_data_loader, test_dataset, extf, epoch=5, name="dnn_upd_", db_range=None):
    global f
    f = open("./models/train_logs/{}_{}e.log".format(name, extf), "w")
    sfe = STFTFeatureExtractor(extf).to(device)
    rfe = RawFeatureExtractor(extf).to(device)
    lfe = LSTMFeatureExtractor(extf).to(device)
    bd = Discriminator(extf).to(device)
    sfe.load_state_dict(dnn_dict["sfe"])
    rfe.load_state_dict(dnn_dict["rfe"])
    lfe.load_state_dict(dnn_dict["lfe"])
    bd.load_state_dict(dnn_dict["bd"])
    optimizer_sfe = torch.optim.Adam(sfe.parameters(), 1e-4, [0.9, 0.999])
    optimizer_rfe = torch.optim.Adam(rfe.parameters(), 1e-4, [0.9, 0.999])
    optimizer_lfe = torch.optim.Adam(lfe.parameters(), 1e-4, [0.9, 0.999])
    optimizer_bd = torch.optim.Adam(bd.parameters(), 1e-4, [0.9, 0.999])
    bd_loss_spec = torch.nn.BCEWithLogitsLoss(reduction="mean")
    print("init DNN===============")
    f.write("init DNN===============")
    nn_decoder_eval(sfe, rfe, lfe, bd, extf, test_dataset, db_range)

    step = 0
    ber_sum = []
    steps = []
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
            if step % 100 == 0:
                s = "dnn, epoch:{}, step:{}, loss:{:.4f}".format(e, step, bd_loss.item())
                print("====================")
                print(s)
                if f is not None:
                    f.write("====================\n")
                    f.write(s + "\n")
                if step > 200:
                    bers = nn_decoder_eval(sfe, rfe, lfe, bd, extf, test_dataset, db_range)
                    ber_sum.append(np.sum(bers))
                    steps.append(step)
                    d = {"sfe": sfe.state_dict(), "rfe": rfe.state_dict(), "lfe": lfe.state_dict(), "bd": bd.state_dict()}
                    torch.save(d, "./models/{}_{}e_{}s.pt".format(name, extf, step))
    min_ber_idx = np.argmin(ber_sum)
    f.write("min_step={}\n".format(steps[min_ber_idx]))
batch_size = 64

best_dnn_round = {1:5000, 2: 6800, 4: 7000, 8: 6800, 16:6400, 32:6400, 64:3400}
def danf_train_main(extf, synthesis):
    dataset_prefix = "white_noise_{}e_".format(extf)
    target_prefix = "wifi_int_{}e".format(extf)
    save_name = "danf"
    source_dataset = ble_data_loader.ble_raw_dataset(dataset_prefix + "train", extf, [-20, 0])
    if synthesis:
        target_dataset = ble_data_loader.ble_raw_dataset(target_prefix + "train", extf, [-20, 0])
        test_dataset = ble_data_loader.ble_raw_dataset(target_prefix + "test", extf, -20)
    else:
        target_dataset = ble_data_loader.ble_true_dataset(target_prefix, extf)
        test_dataset = ble_data_loader.ble_true_dataset(target_prefix, extf)
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

# WR 应该是widerange的意思，之前做距离实验的时候使用[-20, 0]训练的，但是在距离实验时发现近距离反而性能差于原来的，一开始怕过拟合扩大了db到[-20, 5],但是还是需要更新一下才能有更好的性能
def dnn_train_main(extf, name="dnn"):
    dataset_prefix = "white_noise_{}e_".format(extf)
    train_dataset = ble_data_loader.ble_raw_dataset(dataset_prefix + "train", extf, [-20, 20])
    test_dataset = ble_data_loader.ble_raw_dataset(dataset_prefix + "test", extf, -10, True)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    # test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
    dnn_train(train_data_loader, test_dataset, extf, 5, name=name)
    if f is not None:
        f.flush()
        f.close()

def dnn_update_main(extf):
    dataset_prefix = "wifi_int_{}e_".format(extf)
    train_dataset = ble_data_loader.ble_true_dataset(dataset_prefix + "train", extf)
    test_dataset = ble_data_loader.ble_true_dataset(dataset_prefix + "test", extf, fast=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    dnn_dict = torch.load("./models/dnn_{}e_{}s.pt".format(extf, best_dnn_round[extf]))
    dnn_update(dnn_dict, train_dataloader, test_dataset, extf)

def dnn_wifi_update_main(extf):
    # best_dnn_round = {1:6400, 2: 6400, 4: 6800, 8: 5600, 16:6600, 32:6800, 64:3400}
    if extf == 64:    
        SEED = 22
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
    best_dnn_round = {1:3800, 2: 5600, 4: 3400, 8: 6600, 16:5800, 32:6600, 64:3200}
    dataset_prefix = "wifi_int_new_{}e_".format(extf)
    train_dataset = ble_data_loader.ble_true_dataset(dataset_prefix + "train", extf)
    if extf <= 16:
        train_dataset.set_snr([-20, 0])
    else:  
        train_dataset.set_snr([-30, 0])
    test_dataset = ble_data_loader.ble_true_dataset(dataset_prefix + "test", extf, fast=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    dnn_dict = torch.load("./models/dnn_uni_cpy_wr_{}e_{}s.pt".format(extf, best_dnn_round[extf]))
    dnn_update(dnn_dict, train_dataloader, test_dataset, extf, name="dnn_uni_cpy_wifi", epoch=3)

def dnn_wr_update_main(extf):
    syms = []
    labels = []
    dis = []
    test_syms = []
    test_labels = []
    test_dis = []
    ds = np.array([5, 10, 20, 30, 40, 50, 70, 90])

    for d in ds:
        best_steps = {1: 5000, 2: 5600, 4: 3600, 8: 4000, 16: 4000, 32: 4200, 64:3400}
        data = np.load("./processed_data/case_study/outdoor/{}m_18c_{}e_100n_0.npz".format(d, extf))
        sym = data["sym"]
        label = data["label"]
        num = int(0.05 * sym.shape[0])
        syms.append(sym[:num, :].copy())
        labels.append(label[:num].copy())
        dis.append(np.array([d for _ in range(num)]))
        test_syms.append(sym[num:2*num, :].copy())
        test_labels.append(label[num:2*num].copy())
        test_dis.append(np.array([d for _ in range(test_syms[-1].shape[0])]))
        del data
    syms = np.vstack(syms)
    labels = np.concatenate(labels).astype(np.float32)
    dis = np.concatenate(dis)
    test_syms = np.vstack(test_syms)
    test_labels = np.concatenate(test_labels).astype(np.float32)
    test_dis = np.concatenate(test_dis)
    train_dataset = ble_data_loader.ble_ram_dataset(syms, labels, extf, dis)
    test_dataset = ble_data_loader.ble_ram_dataset(test_syms, test_labels, extf, test_dis)
    dnn_dict = torch.load("./models/dnn_wr_{}e_{}s.pt".format(extf, best_steps[extf]))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    dnn_update(dnn_dict, train_dataloader, test_dataset, extf, name="dnn_wr_upd", db_range=ds)

def dnn_uni_update_main(extf):
    syms = []
    labels = []
    dis = []
    test_syms = []
    test_labels = []
    test_dis = []
    ds = np.array([5, 10, 20, 30, 40, 50, 70, 90])
    best_steps = {1:3800, 2: 5600, 4: 3400, 8: 6600, 16:5800, 32:6600, 64:3200}
    
    for d in ds:
        data = np.load("./processed_data/case_study/outdoor/{}m_18c_{}e_100n_0.npz".format(d, extf))
        sym = data["sym"]
        label = data["label"]
        num = int(0.05 * sym.shape[0])
        syms.append(sym[:num, :].copy())
        labels.append(label[:num].copy())
        dis.append(np.array([d for _ in range(num)]))
        test_syms.append(sym[num:2*num, :].copy())
        test_labels.append(label[num:2*num].copy())
        test_dis.append(np.array([d for _ in range(test_syms[-1].shape[0])]))
        del data
    syms = np.vstack(syms)
    labels = np.concatenate(labels).astype(np.float32)
    dis = np.concatenate(dis)
    test_syms = np.vstack(test_syms)
    test_labels = np.concatenate(test_labels).astype(np.float32)
    test_dis = np.concatenate(test_dis)
    train_dataset = ble_data_loader.ble_ram_dataset(syms, labels, extf, dis)
    test_dataset = ble_data_loader.ble_ram_dataset(test_syms, test_labels, extf, test_dis)
    dnn_dict = torch.load("./models/dnn_uni_cpy_wr_{}e_{}s.pt".format(extf, best_steps[extf]))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    dnn_update(dnn_dict, train_dataloader, test_dataset, extf, name="dnn_uni_cpy_wr_dis", db_range=ds)
    
"""
dnn [-20 0] utils.awgn没有copy
dnn_wr [-20, 5]utils.awgn没有copy
dnn_uni [-20, 5]utils.awgn没有copy
dnn_uni_cpy [-20, 0]utils.awgn有copy
dnn_uni_cpy_wr [-20, 10]utils.awgn有copy, 因为之前有那个内存泄露问题，会导致后面的epoch的实际snr会叠加之前的信号，snr上升

"""
if __name__ == "__main__":
    db_ranges = []
    # 对于训练64，需要一点随机性，这边的numpy随机种子固定为11，一般前一次跑的增益很差，但是如果连续跑两次后面那个随机状态变化，反而更好
    # [1, 2, 4, 16, 32, 64]
    for extf in [1, 2, 4, 16, 32, 64]:
        print("+++++++++++++++++++++++++++++++")
        print(extf)
        # dnn_train_main(extf, "dnn_uni_cpy_wr20")
        dnn_uni_update_main(extf)  # 64要重拍跑了

        """
        WiFi 这个跑的时候，1, 2, 4, 16, 32, 64，然后64单独跑
        dnn_wifi_update_main(extf)
        """
        print("+++++++++++++++++++++++++++++++")
        
        # dnn_wr_update_main(extf)
        # danf_train_main(extf, synthesis=False)
        # dnn_update_main(extf)
