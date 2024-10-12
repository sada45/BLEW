# [SenSys 24] Combating BLE Weak Links with Adaptive Symbol Extension and DNN-based Demodulation

This is the articact evaluation code for [SenSys 24] Combating BLE Weak Links with Adaptive Symbol Extension and DNN-based Demodulation

Dataset and trained models is uploaded to [here](https://1drv.ms/f/c/92164f3e5a5e1519/EmU-4IQgTbZLpu0rtWMh_wcBfdj31nx-JRUXIbSKP4vOcQ?e=iHe9K5)

**Hardware setup:** AMD Ryzen 9 7950X and RTX 3090Ti. The runtime experiment uses a laptop with i7-10750H and 1650Ti

**Python Env.:** python 3.11.0, numpy, 1.26.4, scipy 1.13.1, torch 2.3.1,  Driver Version: 535.183.01, CUDA Version: 12.2

## Before start
An absolute path should be updated since code files in different path can uses that code. 
In `blong_dnn.py` Line 443 and 445, change `/liymdata/liym/BLong` to your path store the code.

We first introduce the code structure of BLEW. In the root directory, the `models` includes the trained models and the tranning logs. `output` includes the processed data for ploting figures and analyzing. `processed_data` saves the processed data (e.g., detected packets, trainning datasets, etc.). `raw_data` includes the raw signal files. `scripts` including all the experiment codes.

Sepcifically, the root of `scripts` contains the main code to run the experiments. `scripts/DANF` includes all the DNN trainning/updating and the model components. `scripts/data_collection` includes the code for sampling data with USRP and packet detection. `scripts/opt` incldues the optimization code to choose the best extension factor. 

## DNN trainning and updating
The trainning and updating of the DNN is including in the `scripts/DANF/blong_main_uni.py`. 
`dnn_train_main()` function is used to train new models with the white noise data. 
`dnn_wifi_update_main()` function is used to update the DNN models with the WiFi intereference. 
`dnn_uni_update_main()`  function is used to update the DNN models with the data from a LOS scenario. 

## 7.1 Overall Performance in paper
The code is `scripts/exp_code_nlos_sensys.ipynb`.
We have run the experiment and save the output.
Since we directly put a long signal to demodulate, the total time from raw data is long. 
We saved the intermediate results in `processed_data/`, you can set the `load` in `raw_data_processing` and `get_real_throughput` to `True` to uses processed data. If you requires any raw data, please contact me via liymemnets@zju.edu.cn.
Specifically, (1) run first block to import the header. (2) run second block for generate the throughput of BLEW. The reslut shoule start with `throughput:`, then the position, throughput, preamble detection rate, and the BER. (3) run the thrid block to generate the throughput if Symphony and native BLE. The output should be `<position> tensor(<throughput>)`. The output first shows the throughput of Symphony and then the native BLE.
The position of the output and the actual position in paper has this mapping relationship:
| Output position | Paper position |
|------------|------------|
| C| A|
| D| B|
| F| C|
| G| D|
| H| E|
| I| F|

## 7.2 Impacts of Different Components in paper
For packet detection, the code is located in `scripts/preambel_detection_sensys.ipynb`.
There are two blocks of code. 
The first one generate the `filtered_packet_detection`, the second generate the `awgn_packet_detection`.
Ths second is for the experiment in 7.2, that independently add noise to I and Q path as we claimed in the paper.
The output should be saved in `./output/packet_detection/`, start with `filtered_packet_detection_<extension factor>.csv` or `awgn_packet_detection_<extension factor>.csv`.
The `filtered_packet_detection_<extension factor>.csv` is used to plot the Fig. 11
The `awgn_packet_detection_<extension factor>.csv` is for 7.4(1) energy consumption, we unify the SNR control method with the DNN (AWGN).

For DNN-based demodulator, the code in the second block of `scripts/dnn_gain_exp_sensys.ipynb`.
It output the BER-SER threshold.
Specifaclly, (1) run first block to define the functions. (2->Fig. 13) run second block to output the SNR threshold of DNN. The result stored in `output/snr_relation/cpy_extf=<extension factor>.csv`. The output number is the comparesion with STFT-based (first line) and the native BLE (second line). The list shows the DNN gain with each extension factor, and the number at the end shows the maximum and minimum number. 
(3->Fig. 16) run thrid block to evaluate the DNN with WiFi interference. The first colum is the SINR threshold of non-updated DNN for different extension factors, the second one is for the updated DNN, third for STFT-based, and fourth for the native BLE. 

For the gain of symbol extension (Fig. 14), corresponding to the first block in the code in `scripts/get_snr_threshold.ipynb`.
The code output the SNR threshold of BER and packet detectoon rates with extenshon factor.

## 7.3 Impact of Interference Types in paper
For the BER in outdoor LOS scenario (Fig. 15), the code is in `scripts/exp_code_outdoor.ipynb`. 
It output the BER of updated and original models with different distance.
Specifically, just run the first and second block to get the BER with distance. 
The result stored in `./output/distance_ber_extf=8.csv` is the updated model (firt colum) with distance, and `./output/distance_ber_woupd_extf=8.csv` shows the non-updated model (first colum) with distance.

For WiFi interference, the code in the third block of `scrips/dnn_gain_exp_sensys.ipynb`.

## 7.4(1) BLEW transmitter energy consumption in paper
The code is in `scripts/tx_energy_exp.py`.
The data stored in `./output/energy_consumption.csv`, the second colum is the energy consumption of BLEW, the third is for Symphony, and fourth for the native BLE.