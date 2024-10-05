import sys
sys.path.aooend("..")
import uhd
import numpy as np
import config
import threading
import time 
from scramble_table import *
import cv2
from scripts.data_collection.node_ctrl import node_ctrl
import os
import goodput_opt as gopt
import matplotlib.pylab as plt

# buffer = ringbuffer(config.num_samps, np.complex64)
power_of_2 = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
target_aa_bit = np.array([0,0,0,1,1,0,1,1,0,1,1,1,1,1,0,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1], dtype=np.uint8)
preamble = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
correct_bytes = bytearray.fromhex("d6be898e4623ced55e75fddc02010619096e696d626c655f6c326361705f746573745f736572766572711a85")


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

def raw_signal_segment(data, timestamp, extf):
    i0 = np.real(data[:-1])
    q0 = np.imag(data[:-1])
    i1 = np.real(data[1:])
    q1 = np.imag(data[1:])
    phase_diff = i0 * q1 - i1 * q0
    phase_len = len(phase_diff)
    bits = np.zeros(int(np.ceil(phase_len / config.sample_pre_symbol)), dtype=np.uint8)
    segment_len = extf * 8 + 18 + 32 + 24 + 1 + 16 * extf + 20
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
                # empty_idx = end_idx + 10 * config.sample_pre_symbol
                # empty_end_idx = empty_idx + config.sample_pre_symbol * segment_len + config.sample_pre_symbol
                # if idx >=0 and empty_end_idx < len(data):
                if idx >= 0:
                    # bits_data = bits[preamble_idx[j]: preamble_idx[j] + 166 * 8].reshape(-1, 8)
                    # bytes_data = np.sum(bits_data * power_of_2, 1, dtype=np.uint8)
                    # bytes_data[4: 6] ^= scramble_table[8][:2]
                    # bytes_data = bytearray(bytes_data)
                    # print(bytes_data.hex())
                    raw_data = data[idx: end_idx]
                    # empty_raw_data = data[empty_idx: empty_end_idx]
                    # print(idx, end_idx)
                    # return raw_data, empty_raw_data, timestamp[0] + idx * (1 / config.sample_rate)
                    return raw_data.copy(), timestamp[0] + idx * (1 / config.sample_rate)
                else:
                    print("sample too short", i, idx, end_idx)
    print("not find")
    return None, -1

class usrp_control(threading.Thread):
    def __init__(self, usrp_type, chan, extf, prefix):
        threading.Thread.__init__(self)
        self.n1 = uhd.usrp.MultiUSRP("type=" + usrp_type)
        self.n1.set_rx_rate(config.sample_rate, 0)
        self.n1.set_rx_freq(uhd.libpyuhd.types.tune_request(get_freq_with_chan_num(chan)), 0)
        self.n1.set_rx_gain(config.gain, 0)
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [0]
        self.metadata = uhd.types.RXMetadata()
        self.streamer = self.n1.get_rx_stream(st_args)
        self.recv_buffer = np.zeros((1, config.num_samples_each), dtype=np.complex64)
        self._is_streaming = True
        self.sample = np.zeros((config.repeat_times, config.num_samples), dtype=np.complex64)
        self.time_stamp = np.zeros((config.repeat_times, config.num_samples // config.num_samples_each), dtype=np.float64)
        self.stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        self.stream_cmd.stream_now = True
        self.stream_cmd.num_samps = config.num_samples_each
        self.raw_data_segments = []
        self.raw_data_timestamp = []
        self.raw_data_index = []
        self.raw_empty_data_segments = []
        # two array to store
        self.raw_signal_arr = None
        self.time_stamp_arr = None 
        self.extf = extf
        if self.extf != -1:
            self.prefix = prefix + "_chan=" + str(chan) + "_extf=" + str(extf)
        else:
            self.prefix = prefix + "_chan=" + str(chan)

    def start_stream(self, sample_num=config.repeat_times):
    # Set up the stream and receive buffer
        # Start Stream
        self.streamer.issue_stream_cmd(self.stream_cmd)
        time_diff = (time.time_ns() // 1e9) - self.n1.get_time_now().get_real_secs()
        sample_index = 0

        while sample_index < sample_num:
            for i in range(config.num_samples // config.num_samples_each):
                # Receive Samples
                s_len = self.streamer.recv(self.recv_buffer, self.metadata)
                start_timestamp = self.metadata.time_spec.get_real_secs()
                self.time_stamp[sample_index, i] = start_timestamp + time_diff
                self.sample[sample_index, config.num_samples_each*i: config.num_samples_each*(i+1)] = self.recv_buffer[0]
            sample_index += 1
            # raw_data, timestamp = raw_signal_segment(self.sample[sample_index, :], self.time_stamp[sample_index, :])
            # if raw_data is not None:
            #     self.raw_data_segments.append(raw_data)
            #     self.raw_data_timestamp.append(timestamp)
            #     # print(self.time_stamp[sample_index, :])
            #     print(self.time_stamp[sample_index, 1:] - self.time_stamp[sample_index, :-1])
            #     sample_index += 1

    def stop_stream(self, sample_num=config.repeat_times):
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        self.streamer.issue_stream_cmd(stream_cmd)
        if self.extf == -1:
            self.raw_signal_arr = self.sample[:, :100000]
            self.time_stamp_arr = self.time_stamp
            
        else:
            for i in range(sample_num):
                raw_signal, signal_time_stamp = raw_signal_segment(self.sample[i, :], self.time_stamp[i, :], self.extf)
                if raw_signal is not None:
                    self.raw_data_segments.append(raw_signal)
                    self.raw_data_timestamp.append(signal_time_stamp)
                    self.raw_data_index.append(i)
                    # self.raw_empty_data_segments.append(empty_raw_signal)
                else:
                    print(i)
            # raw_data_array = np.array(self.raw_data_segments)
            # raw_empty_data_array = np.array(self.raw_empty_data_segments)
            # time_stamp_array = np.array(self.raw_data_timestamp)
        # print(raw_data_array.shape)
        # if len(raw_data_array.shape) != 2 or raw_data_array.shape[1] != (self.extf * 8 + 18 + 32 + 24 + 1 + 16 * self.extf + 20) * config.sample_pre_symbol:
        #     return -1
        
        # return raw_data_array.shape[0]
        # print(self.time_stamp[:, 1:] - self.time_stamp[:, :-1])
        
    
    def run(self):
        # while self._is_streaming:
        self.start_stream()
        self.stop_stream()

        if self.extf != -1:
            while True:
                while len(self.raw_data_segments) < config.repeat_times:
                    self.start_stream(config.repeat_times - len(self.raw_data_segments))
                    self.stop_stream(config.repeat_times - len(self.raw_data_segments))
                self.raw_signal_arr = np.array(self.raw_data_segments)
                self.time_stamp_arr = np.array(self.raw_data_timestamp)
                if len(self.raw_signal_arr.shape) != 2:
                    self.raw_data_segments = []
                    self.raw_data_timestamp = []
                    self.raw_data_index = []
                else:
                    break       
        np.savez(self.prefix + ".npz", self.raw_signal_arr, self.time_stamp_arr)

    def stop(self):
        self._is_streaming = False



def get_chan_raw(train):
    nc = node_ctrl()
    # 2, 4, 16, 32, 64, 128
    for extf in [128]:
        for chan in range(40):
            while True:
                res = 0
                res += nc.stop()
                res += nc.set_chan(chan)
                res += nc.set_extend_factor(extf)
                res += nc.start()
                if res < 0:
                    print("chan ", chan, "error, retrying")
                    time.sleep(1)
                else:
                    break
            for i in range(1):
                print("./raw_data/single_chan/" + "chan=" + str(chan) + "_extf=" + str(extf) + "_" + str(i), "train=", train)
                if train:
                    rx_thread = usrp_control("b200, serial=8001044", chan, extf, "./raw_data/data")
                else:
                    rx_thread = usrp_control("b200, serial=8001044", chan, extf, "./raw_data/data_test")
                rx_thread.daemon = False
                rx_thread.start()
                rx_thread.join()
                del rx_thread
    del nc

def get_white_noise(train=True):
    nc = node_ctrl()
    for chan in range(40):
        rc = nc.stop()
        if rc != 0:
            return 
        for i in [1]:
            print("./raw_data/" + "chan=" + str(chan) + "_" + str(i))
            if train:
                rx_thread = usrp_control("b200, serial=8001044", chan, -1, "./raw_data/white_noise_" + str(i))
            else:
                rx_thread = usrp_control("b200, serial=8001044", chan, -1, "./raw_data/white_noise_test_" + str(i))
            
            rx_thread.daemon = False
            rx_thread.start()
            rx_thread.join()
            del rx_thread
    del nc

def get_lost_file_names(chans, extfs, train):
    file_names = os.listdir("./raw_data/")
    for chan in chans:
        for extf in extfs:
            if train:
                target_file_name = "data_chan=" + str(chan) + "_extf=" + str(extf) + ".npz"
            else:
                target_file_name = "data_test_chan=" + str(chan) + "_extf=" + str(extf) + ".npz"
            if target_file_name not in file_names:
                print(target_file_name)


def auto_upload():
    files = []
    while True:
        new_files = []
        for filename in os.listdir("./raw_data/single_chan"):
            if filename not in files and filename.find("white_noise") != -1:
                new_files.append(filename)
        for file in new_files:
            os.system("scp ./raw_data/single_chan/" + file + " gpu-3@10.214.131.232:/home/gpu-3/liym/BLong/BLong_nn/raw_data/single_chan/ > /dev/null")
            files.append(file)
        time.sleep(60)



def throughput_raw_data():
    nc = node_ctrl()
    for snr in range(-12, 7, 2):
        n_max, extf = gopt.goodput_optimization(snr)
        for chan in range(40):
            while True:
                res = 0
                res += nc.stop()
                res += nc.set_chan(chan)
                res += nc.set_extf_nmax(extf, n_max)
                res += nc.start()
                if res < 0:
                    time.sleep(1)
                else:
                    break
            rx_thread = usrp_control("b200, serial=8001044", chan, extf, "./raw_data/goodput/th_")
            rx_thread.daemon = False
            rx_thread.start()
            rx_thread.join()
            del rx_thread
    del nc

def get_1m_power():
    nc = node_ctrl()
    while True:
        res = 0
        res += nc.stop()
        res += nc.set_chan(0)
        res += nc.set_extf_nmax(8, 5)
        res += nc.start()
        if res < 0:
            time.sleep(1)
            print(12321321)
        else:
            break
    # 2 is for outdoor
    rx_thread = usrp_control("b200, serial=8001044", 0, 8, "./raw_data/power_1m_2")
    rx_thread.daemon = False
    rx_thread.start()
    rx_thread.join()
    del rx_thread
    del nc


# if __name__ == "__main__":
#     get_chan_raw(False)
#     get_chan_raw(True)

get_1m_power()
# rx_thread = usrp_control("b200, serial=8001044", 0, -1, "./raw_data/power_empty_3")
# rx_thread.daemon = False
# rx_thread.start()
# rx_thread.join()
# del rx_thread