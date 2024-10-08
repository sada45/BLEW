# [SenSys 24] Combating BLE Weak Links with Adaptive Symbol Extension and DNN-based Demodulation

This is the articact evaluation code for [SenSys 24] Combating BLE Weak Links with Adaptive Symbol Extension and DNN-based Demodulation

Dataset and trained models is uploaded to [here](https://1drv.ms/f/c/92164f3e5a5e1519/EmU-4IQgTbZLpu0rtWMh_wcBfdj31nx-JRUXIbSKP4vOcQ?e=iHe9K5)

**Hardware setup:** AMD Ryzen 9 7950X and RTX 3090Ti. The runtime experiment uses a laptop with i7-10750H and 1650Ti

**Python Env.:** python 3.11.0, numpy, 1.26.4, scipy 1.13.1, torch 2.3.1,  Driver Version: 535.183.01, CUDA Version: 12.2

## 7.1 Overall Performance
The code is `scripts/exp_code_nlos_sensys.ipynb`.
We have run the experiment and save the output.
Since we directly put a long signal to demodulate, the total time from raw data is long. 
We saved the intermediate results in `processed_data/`, you can set the `load` in `raw_data_processing` and `get_real_throughput` to `True` to uses processed data. If you requires any raw data, please contact me via liymemnets@zju.edu.cn.

## 7.2 Impacts of Different Components
For packet detection, the code is located in `scripts/preambel_detection_sensys.ipynb`.
There are two blocks of code. 
The first one generate the `awgn_packet_detection`, the second generate the `filter_packet_detection`.
Ths second is for the experiment in 7.2, that independently add noise to I and Q path as we claimed in the paper.
The first is for 7.4(1) energy consumption, we unify the SNR control method with the DNN (AWGN).

For DNN-based demodulator, the code in the second block of `scripts/dnn_gain_exp_sensys.ipynb`.
It output the BER-SER threshold.

For the gain of symbol extension, the code in `scripts/get_snr_threshold.ipynb`.
The code output the SNR threshold of BER and packet detectoon rates with extenshon factor.

## 7.3 Impact of Interference Types
For the BER in outdoor LOS scenario, the code is in `scripts/exp_code_outdoor.ipynb`. 
It output the BER of updated and original models with different distance.

For WiFi interference, the code in the third block of `scrips/dnn_gain_exp_sensys.ipynb`.

## 7.4(1) BLEW transmitter energy consumption
The code is in `scripts/tx_energy_exp.py`