import torch 
import numpy as np
from scripts.DANF.blong_nn_components import *

m = 1024 * 1024

best_model_list = ["blong_nn_extf=1_round=0_final.pt", "blong_nn_extf=2_round=0_final.pt", "blong_nn_extf=4_round=0_final.pt","blong_nn_extf=8_round=2_final_0.pt", "blong_nn_extf=16_round=4_final_0.pt", "blong_nn_extf=32_round=3_final.pt", "blong_nn_extf=64_round=3_final.pt", "blong_nn_extf=128_round=4_final.pt"]

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size)
    
    return param_size / m

for extf in [1, 2, 4, 8, 16, 32, 64, 128]:
    model_name = best_model_list[int(np.log2(extf))]
    model_states = torch.load("./models/models/" + model_name)
    fe = FeatureExtractor()
    nf = NoiseFilter(extf)
    cd = ChannelDiscriminator(extf)
    fe.load_state_dict(model_states["fe_state"])
    nf.load_state_dict(model_states["nf_state"])
    print(extf, getModelSize(fe)+getModelSize(nf), getModelSize(cd))
    