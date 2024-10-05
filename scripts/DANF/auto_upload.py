import time
import os

files = []
while True:
    new_files = []
    for filename in os.listdir("./raw_data/"):
        if filename not in files and filename.find("data") != -1:
            new_files.append(filename)
    for file in new_files:
        os.system("scp ./raw_data/" + file + " gpu-3@10.214.131.232:/home/gpu-3/liym/BLong/BLong_nn/raw_data/")
        files.append(file)
    time.sleep(60)