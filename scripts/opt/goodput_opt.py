import numpy as np
# import matplotlib.pyplot as plt
# We modified the pwlf lib
import sys
sys.path.append("./scripts/")
import blong_pwlf as pwlf
import time
from opt.ber_relations import *

plf = None
plf_list = []
plf_betas = None
plf_breaks = None
extf_list = [1, 2, 4, 8, 16, 32, 64, 128]
n_max_list = [n for n in range(8, 249, 8)] + [255]
extf_num = 8
N_f = 10
blong_bers = None
dnn_bers = None
stft_bers = None
ber_bound = []

def get_snr_ber_relationship(extf):
    blong_acc = np.zeros(41, dtype=np.float64)
    dnn_acc = np.zeros(41, dtype=np.float64)
    stft_acc = np.zeros(41, dtype=np.float64)
    with open("./logs/eval_test_extf=" + str(extf) + ".txt", 'r') as f:
        lines = f.readlines()
        idx = 0
        while lines[idx].find("=========") == -1:
            idx += 1
        idx += 1
        for i in range(41):
            line = lines[idx+i]
            line = line[line.find("=")+1:]
            segs = line.split(",")
            blong_acc[i] = float(segs[0])
            dnn_acc[i] = float(segs[1])
            stft_acc[i] = float(segs[2])
    return 1 - blong_acc, 1 - dnn_acc, 1 - stft_acc

def get_native_snr_ber_relationship():
    acc = []
    for extf in extf_list:
        stft_acc = np.zeros(41, dtype=np.float64)
        with open("./logs/native_stft_without_hanning_extf=" + str(extf) + ".txt", 'r') as f:
            lines = f.readlines()
            for i in range(41):
                line = lines[i]
                line = line[line.find("=")+1:]
                stft_acc[i] = float(line)
        acc.append(1 - stft_acc)
    return np.array(acc)


def load_snr_ber_relationship():
    data = np.load("./export_data/bers.npz")
    return data["blong_bers"], data["dnn_bers"], data["stft_bers"]

def plf_fitting(snr_ber, db=None):
    if db is None:
        db = np.arange(-30, 11, 1, dtype=np.float64)
    plf = pwlf.PiecewiseLinFit(db, snr_ber, seed=11, blong=True)
    # The SNR-BER is segmented into three parts
    res = plf.fit(3)
    # x_pre = np.linspace(-30, 10, num=10000)
    # y_pre = plf.predict(x_pre)
    # origin_y = plf.predict(db)
    return plf

def get_ber(snr_val):
    snr = np.array([snr_val], dtype=np.float64)
    # We have 9 options for n_ext
    ber = np.zeros(extf_num, dtype=np.float64)
    for i in range(extf_num):
        # ber[i] = plf_list[i].predict(snr)
        ber[i] = plf.predict(snr, plf_betas[i, :], plf_breaks[i, :])
    # To reduce the fitting error, when on the 3rd segment, we set the BER to the floor
    ber[np.where(ber<=ber_bound[1])] = ber_bound[0]
    return ber

def cal_pdr(n_max, snr_val):
    ber = get_ber(snr_val).reshape(1, -1)
    ber = np.tile(ber, (n_max.shape[0], 1))
    # len(n_max) * extf_num
    pdr = np.power(1-ber, (n_max+N_f)*8)
    return pdr

def goodput_optimization(snr_val, n_max=np.array(n_max_list)):
    n_max = n_max.reshape(-1, 1)
    pdr = cal_pdr(n_max, snr_val)
    np.savetxt("./figs/pdr.csv", pdr, delimiter=",")
    goodput = np.zeros((n_max.shape[0], extf_num), dtype=np.float64)
    # For extension factor = 1, we reuse the native fields in the native AIPs
    goodput[:, 0:1] = n_max / (n_max + N_f + 150 / 8) * pdr[:, 0:1]
    # Otherwise, N_f * n_ext extra length needs to be transmitted
    extf_arr = np.array(extf_list[1:]).reshape(1, -1)
    n_total = (n_max + N_f) * extf_arr
    n_bit = np.floor(255 * 8 / extf_arr)
    n_aip = np.ceil(n_total * 8 / n_bit)
    goodput[:, 1:] = (n_max * pdr[:, 1:]) / (n_total + (N_f + 150 / 8) * n_aip)
    max_idx = np.unravel_index(goodput.argmax(), goodput.shape)
    return n_max[max_idx[0], 0], extf_list[max_idx[1]], goodput[max_idx]

def save_bers():
    blong_bers = []
    dnn_bers = []
    stft_bers = []
    for extf in extf_list:
        blong_ber, dnn_ber, stft_ber = get_snr_ber_relationship(extf)
        blong_bers.append(blong_ber)
        dnn_bers.append(dnn_ber)
        stft_bers.append(stft_ber)
    blong_bers = np.array(blong_bers)
    dnn_bers = np.array(dnn_bers)
    stft_bers = np.array(stft_bers)
    np.savez("./export_data/bers.npz", blong_bers=blong_bers, dnn_bers=dnn_bers, stft_bers=stft_bers)

def fit_and_save_plf():
    global blong_bers, dnn_bers, stft_bers
    blong_bers, dnn_bers, stft_bers = load_snr_ber_relationship()
    db = np.arange(-30, 11, 1, dtype=np.float64)
    for i in range(len(extf_list)):
        plf = pwlf.PiecewiseLinFit(db, blong_bers[i, :], seed=11, blong=True)
        plf.fit(3)
        np.savez("./plf_params/plf_extf=" + str(extf_list[i]) + ".npz", breaks=plf.fit_breaks, beta=plf.beta)

def plf_init(test_thres=False):
    global blong_bers, dnn_bers, stft_bers, plf, plf_betas, plf_breaks, ber_bound
    blong_bers, dnn_bers, stft_bers = load_snr_ber_relationship()
    stft_bers = get_native_snr_ber_relationship()
    plf_betas = []
    plf_breaks = []
    plf = pwlf.PiecewiseLinFit(np.arange(-30, 11, 1), blong_bers[0, :], seed=11, blong=True)
    test_snr = np.array([100])
    ber_floors = np.zeros(len(extf_list))
    for i in range(len(extf_list)):
        param = np.load("./plf_params/plf_extf=" + str(extf_list[i]) + ".npz")
        plf_betas.append(param["beta"])
        plf_breaks.append(param["breaks"])
        ber_floors[i] = plf.predict(test_snr, param["beta"], param["breaks"])[0]
        print(extf_list[i], plf_breaks[-1][2] - plf_breaks[-1][1])
    plf_betas = np.array(plf_betas)
    plf_breaks = np.array(plf_breaks)
    ber_bound = [np.min(ber_floors), np.max(ber_floors)]
    if test_thres:
        for i in range(len(extf_list)):
            p2 = pwlf.PiecewiseLinFit(np.arange(-30, 11, 1), dnn_bers[i], seed=11, blong=True)
            p2.fit(3)
            p3 = pwlf.PiecewiseLinFit(np.arange(-30, 11, 1), stft_bers[i], seed=11, blong=True)
            p3.fit(3)
            beta = [plf_betas[i], p2.beta, p3.beta]
            breaks = [plf_breaks[i], p2.fit_breaks, p3.fit_breaks]
            thres_db = np.zeros(3)
            for j in range(3):
                thres = (0.1 - beta[j][0] + breaks[j][0] * beta[j][1] + breaks[j][1] * beta[j][2]) / (beta[j][1] + beta[j][2])
                thres_db[j] = thres
            print(thres_db)



# The code for experiments in real world
def get_ber_distance(distance, scenario):
    bers = np.zeros(7)
    if scenario == "indoor":
        for i in range(7):
            extf = int(2 ** i)
            bers[i] = indoor_bcrs[distance][extf]
    elif scenario == "outdoor":
        for i in range(7):
            extf = int(2 ** i)
            bers[i] = outdoor_bcrs[distance][extf]
    else:
        raise Exception("No such a scenario")
    return 1 - bers

def get_ber_snr(snr):
    bers = np.zeros(7)
    for i in range(7):
        extf = int(2 ** i)
        bers[i] = snr_ber_relation[snr][extf]
    return 1 - bers

def goodput_optimiation_distance(distance, scenario, n_max=np.arange(1, 256)):
    extf_list = np.array([1, 2, 4, 8, 16, 32, 64])
    extf_num = len(extf_list)
    n_max = n_max[:, np.newaxis]
    bers = get_ber_distance(distance, scenario)[np.newaxis, :]
    bers = np.tile(bers, (n_max.shape[0], 1))
    pdr = np.power(1-bers, (n_max+N_f)*8)
    goodput = np.zeros((n_max.shape[0], extf_num), dtype=np.float64)
    # For extension factor = 1, we reuse the native fields in the native AIPs
    goodput[:, 0:1] = n_max / (n_max + N_f + 150 / 8) * pdr[:, 0:1]
    # Otherwise, N_f * n_ext extra length needs to be transmitted
    extf_arr = np.array(extf_list[1:]).reshape(1, -1)
    n_total = (n_max + N_f) * extf_arr
    n_bit = np.floor(255 * 8 / extf_arr)
    n_aip = np.ceil(n_total * 8 / n_bit)
    goodput[:, 1:] = (n_max * pdr[:, 1:]) / (n_total + (N_f + 150 / 8) * n_aip)
    max_idx = np.unravel_index(goodput.argmax(), goodput.shape)
    return n_max[max_idx[0], 0], extf_list[max_idx[1]], goodput[max_idx]
# s = "{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(d, extf, plr, bcrs[0], bcrs[1], bcrs[2], cpns[0], cpns[1], cpns[2])

def goodput_optimiation_nlos(bcrs, pdetrs, n_max=np.arange(1, 256)):
    extf_list = np.array([1, 2, 4, 8, 16, 32, 64])
    extf_num = len(extf_list)
    n_max = n_max[:, np.newaxis]
    bers = 1 - bcrs
    pdetrs = pdetrs
    bers = bers[np.newaxis, :]
    bers = np.tile(bers, (n_max.shape[0], 1))
    pdr = np.power(1-bers, (n_max+N_f)*8)
    goodput = np.zeros((n_max.shape[0], extf_num), dtype=np.float64)
    # For extension factor = 1, we reuse the native fields in the native AIPs
    goodput[:, 0:1] = n_max / (n_max + N_f + 150 / 8) * pdr[:, 0:1]
    # Otherwise, N_f * n_ext extra length needs to be transmitted
    extf_arr = np.array(extf_list[1:]).reshape(1, -1)
    n_total = (n_max + N_f) * extf_arr
    n_bit = np.floor(255 * 8 / extf_arr)
    n_aip = np.ceil(n_total * 8 / n_bit)
    goodput[:, 1:] = (n_max * pdr[:, 1:]) / (n_total + (N_f + 150 / 8) * n_aip)
    goodput = goodput * pdetrs[np.newaxis, :]
    max_idx = np.unravel_index(goodput.argmax(), goodput.shape)
    return n_max[max_idx[0], 0], extf_list[max_idx[1]], goodput[max_idx]

def goodput_optimiation_nlos_fixn(bcrs, pdetrs, n_max=255, pdr_thres=None):
    extf_list = np.array([1, 2, 4, 8, 16, 32, 64])
    extf_num = len(extf_list)
    pdetrs = pdetrs
    bcrs = bcrs[np.newaxis, :]
    goodput = np.zeros(extf_num, dtype=np.float64)
    # For extension factor = 1, we reuse the native fields in the native AIPs
    goodput[0] = n_max / (n_max + N_f + 150 / 8) * bcrs[:, 0:1]
    
    # Otherwise, N_f * n_ext extra length needs to be transmitted
    n_total = (n_max + N_f) * extf_list[1:]
    n_bit = np.floor(255 * 8 / extf_list[1:])
    n_aip = np.ceil(n_total * 8 / n_bit)
    goodput[1:] = (n_max * bcrs[:, 1:]) / (n_total + (N_f + 150 / 8) * n_aip)
    goodput = goodput * pdetrs[:]
    if pdr_thres is not None:
        max_idxs = np.argsort(-goodput)
        for i in max_idxs:
            if pdetrs[i] >= pdr_thres:
                return n_max, extf_list[i], goodput[i]
    else:
        max_idx = np.argmax(goodput)
        return n_max, extf_list[max_idx], goodput[max_idx]
    return None, None, None


def goodput_optimiation_snr_fixn(bcrs, pdetrs, n_max=255):
    extf_list = np.array([1, 2, 4, 8, 16, 32, 64])
    extf_num = len(extf_list)
    goodput = np.zeros(extf_num, dtype=np.float64)
    # For extension factor = 1, we reuse the native fields in the native AIPs
    goodput[0] = n_max / (n_max + N_f + 150 / 8) * bcrs[0]
    # Otherwise, N_f * n_ext extra length needs to be transmitted
    n_total = (n_max + N_f) * extf_list[1:]
    n_bit = np.floor(255 * 8 / extf_list[1:])
    n_aip = np.ceil(n_total * 8 / n_bit)
    goodput[1:] = (n_max * bcrs[1:]) / (n_total + (N_f + 150 / 8) * n_aip)
    goodput = goodput * pdetrs[:]
    max_idx = np.argmax(goodput)
    return n_max, extf_list[max_idx], goodput[max_idx]


def goodput_optimization_snr(snr, n_max=np.arange(1, 256)):
    extf_num = 7
    extf_list = np.array([1, 2, 4, 8, 16, 32, 64])
    n_max = n_max[:, np.newaxis]
    bers = get_ber_snr(snr)[np.newaxis, :]
    bers = np.tile(bers, (n_max.shape[0], 1))
    pdr = np.power(1-bers, (n_max+N_f)*8)
    goodput = np.zeros((n_max.shape[0], extf_num), dtype=np.float64)
    # For extension factor = 1, we reuse the native fields in the native AIPs
    goodput[:, 0:1] = n_max / (n_max + N_f + 150 / 8) * pdr[:, 0:1]
    # Otherwise, N_f * n_ext extra length needs to be transmitted
    extf_arr = np.array(extf_list[1:]).reshape(1, -1)
    n_total = (n_max + N_f) * extf_arr
    n_bit = np.floor(255 * 8 / extf_arr)
    n_aip = np.ceil(n_total * 8 / n_bit)
    goodput[:, 1:] = (n_max * pdr[:, 1:]) / (n_total + (N_f + 150 / 8) * n_aip)
    max_idx = np.unravel_index(goodput.argmax(), goodput.shape)
    return n_max[max_idx[0], 0], extf_list[max_idx[1]], goodput[max_idx]

def get_dis_ber(scenario):
    dic = {}
    with open("./output/" + scenario + "_bcrs.log", "r") as f:
        lines = f.readlines()
        for line in lines:
            bs = line.split(",")
            data = []
            for i in range(len(bs)):
                data.append(float(bs[i]))
            d = int(data[0])
            extf = int(data[1])
            if extf == 1:
                bcr = data[3]
            else:
                bcr = data[2]
            if d not in dic:
                dic[d] = {}
            dic[d][extf] = bcr
    print(dic)

def get_brr():
    for extf in [64]: 
        print(extf, end=",")  
        brr = []
        for d in [10, 20, 30, 50, 70, 90, 110]:
            with open("./output/outdoor_bcrs.log", "r") as f:
                lines = f.readlines()
                for line in lines:
                    bs = line.split(",")
                    data = []
                    for i in range(len(bs)):
                        data.append(float(bs[i]))
                    f_d = int(data[0])
                    f_extf = int(data[1])
                    if f_extf == extf and f_d == d:
                        brr.append(data[3])
        print(brr)




if __name__ == "__main__":
    get_brr()
    # get_dis_ber("outdoor")
    # for d in [5, 10, 15, 20, 25, 30, 40]:
    #     n_max, extf, _ = goodput_optimiation_distance(d, "indoor")
    #     print(d, n_max, extf)




            