import numpy as np
import torch
from blong_nn_components import *
from blong_eval import *
import ble_data_loader
from torch.utils.data import DataLoader
import multiprocessing


f = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# To make sure the model reproducibility
np.random.seed(config.seed)
torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)

def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-10.*p)) - 1.

# def get_lambda(epoch, max_epoch):
#     p = epoch / max_epoch
#     return 2 - 2. / (1+np.exp(-10.*p))

def blong_nn_train(train_data_loader, test_data_loader, extf, epoch=20, lamb=0.1):
    fe = FeatureExtractor().to(device)
    nf = NoiseFilter(extf).to(device)
    cd = ChannelDiscriminator(extf).to(device)
    optimizer_fe = torch.optim.Adam(fe.parameters(), 1e-3, [0.9, 0.999])
    optimizer_nf = torch.optim.Adam(nf.parameters(), 1e-3, [0.9, 0.999])
    optimizer_cd = torch.optim.Adam(cd.parameters(), 1e-3, [0.9, 0.999])
    loss_spec = torch.nn.MSELoss(reduction='mean')
    loss_chan = torch.nn.BCEWithLogitsLoss()

    train_loss_scale = config.loss_scale
    if extf < config.stft_window_size:
        train_loss_scale = config.loss_scale / (config.stft_window_size / extf)
    
    step = 0
    for e in range(epoch):
        l = lamb * get_lambda(e+1, epoch)
        for idx, ((noise_sym, sym, label, _, _), (target_noise_sym, _, _, _, _)) in enumerate(zip(train_data_loader, test_data_loader)):
            sym = sym.to(device)
            noise_sym = noise_sym.to(device)
            target_noise_sym = target_noise_sym.to(device)
            label = label.to(device)
            mixed_sym = torch.cat((noise_sym, target_noise_sym), dim=0)
            # 0 for source domain, 1 for target domain
            domain_label = torch.zeros((mixed_sym.shape[0], 1), dtype=torch.float32).to(device)
            domain_label[noise_sym.shape[0]:, 0] = 1

            # Frist, train the domain discrimnator
            features = fe(mixed_sym)
            predicted_chan_label = cd(features.detach())  # detach since we do not update FE yet
            loss_cd = loss_chan(predicted_chan_label, domain_label)
            cd.zero_grad()
            loss_cd.backward()
            optimizer_cd.step()
            
            # Then we train the feature extractor and noise filter
            denoised_sym = nf(features[0: noise_sym.shape[0]])
            denoised_sym = denoised_sym * noise_sym
            predicted_chan_label = cd(features)  # do not detach since we have to update the FE now
            loss_nf = loss_spec(denoised_sym, sym) / train_loss_scale
            loss_cd = loss_chan(predicted_chan_label, domain_label)
            total_loss =  loss_nf - l * loss_cd
            fe.zero_grad()
            nf.zero_grad()
            cd.zero_grad()
            total_loss.backward()
            optimizer_nf.step()
            optimizer_fe.step()
            step += 1

            if step % 100 == 0:
                s = "blong_nn, epoch:{}, step:{}, NF_loss:{:.4f}, CD_loss:{:.4f}".format(e, step, loss_nf.item(), loss_cd.item())
                print(s)
                f.write(s + "\n")
                f.flush()
            
            # Evaluate the NN in test dataset
            if step % 1000 == 0:
                # model_eval(fe, nf, test_data_loader, "blong_cnn,")
                if step >= 2000:
                    d = {"fe_state": fe.state_dict(), "nf_state": nf.state_dict(), "cd_state": cd.state_dict()}
                    torch.save(d, "./models/blong_nn_extf=" + str(extf) + "_" + str(step) + ".pt")
    # Store the final model 
    d = {"fe_state": fe.state_dict(), "nf_state": nf.state_dict(), "cd_state": cd.state_dict()}
    torch.save(d, "./models/blong_nn_extf=" + str(extf) + "_final.pt")
    return fe, nf

def cnn_train(train_data_loader, test_data_loader, extf, epoch=20):
    fe = FeatureExtractor().to(device)
    nf = NoiseFilter(extf).to(device)
    optimizer_fe = torch.optim.Adam(fe.parameters(), 1e-3, [0.9, 0.999])
    optimizer_nf = torch.optim.Adam(nf.parameters(), 1e-3, [0.9, 0.999])
    loss_spec = torch.nn.MSELoss(reduction='mean')

    train_loss_scale = config.loss_scale
    if extf < config.stft_window_size:
        train_loss_scale = config.loss_scale / (config.stft_window_size / extf)

    step = 0
    for e in range(epoch):
        for idx, (noise_sym, sym, label, chan_label, db) in enumerate(train_data_loader):
            sym = sym.to(device)
            noise_sym = noise_sym.to(device)
            label = label.to(device)
            
            # Train the feature extractor and noise filter
            features = fe(noise_sym)
            denoised_sym = nf(features)
            denoised_sym = denoised_sym * noise_sym
            loss_nf = loss_spec(denoised_sym, sym) / train_loss_scale
            fe.zero_grad()
            nf.zero_grad()
            loss_nf.backward()
            optimizer_nf.step()
            optimizer_fe.step()
            step += 1

            if step % 100 == 0:
                s = "cnn, epoch:{}, step:{}, NF_loss:{:.4f}".format(e, step, loss_nf.item())
                print("cnn, epoch:{}, step:{}, NF_loss:{:.4f}".format(e, step, loss_nf.item()))
                f.write(s + "\n")
                f.flush()
            
            # Evaluate the NN in test dataset
            if step % 1000 == 0:
                # model_eval(fe, nf, test_data_loader, "cnn,")
                if step > 2000:
                    d = {"fe_state": fe.state_dict(), "nf_state": nf.state_dict()}
                    torch.save(d, "./models/cnn_extf=" + str(extf) + "_" + str(step) + ".pt")
    d = {"fe_state": fe.state_dict(), "nf_state": nf.state_dict()}
    torch.save(d, "./models/cnn_extf=" + str(extf) + "_final.pt")
    return fe, nf


"""
Train the BLong_NN and DNN toghter,
since the data only need to be prepocessed once,
it can speed up the trainning 
We find that the dynamic learning rate is quite necessary for both DNN and BLong_NN
"""
init_lr = 0.01
def train_toghter(train_data_loader, test_data_loader, extf,  round=0, blong_epoch=20, cnn_epoch=20, lamb=0.1):
    train_log = open("./logs/train_log_extf=" + str(extf) + ".txt", 'w')
    blong_fe = FeatureExtractor().to(device)
    blong_nf = NoiseFilter(extf).to(device)
    blong_cd = ChannelDiscriminator(extf).to(device)
    cnn_fe = FeatureExtractor().to(device)
    cnn_nf = NoiseFilter(extf).to(device)
    # leanning rate for our NN should be litter bigger since we have to deal with two loss
    blong_optimizer_fe = torch.optim.Adam(blong_fe.parameters(), init_lr, [0.5, 0.9])
    blong_optimizer_nf = torch.optim.Adam(blong_nf.parameters(), init_lr, [0.5, 0.9])
    blong_optimizer_cd = torch.optim.Adam(blong_cd.parameters(), init_lr, [0.5, 0.9])
    milstone_list = [i *  20 // 4 for i in range(1, 4)]
    blong_scheduler_fe = torch.optim.lr_scheduler.MultiStepLR(blong_optimizer_fe, milestones=milstone_list, gamma=0.1)
    blong_scheduler_nf = torch.optim.lr_scheduler.MultiStepLR(blong_optimizer_nf, milestones=milstone_list, gamma=0.1)
    blong_scheduler_cd = torch.optim.lr_scheduler.MultiStepLR(blong_optimizer_cd, milestones=milstone_list, gamma=0.1)
    cnn_optimizer_fe = torch.optim.Adam(cnn_fe.parameters(), init_lr, [0.5, 0.9])
    cnn_optimizer_nf = torch.optim.Adam(cnn_nf.parameters(), init_lr, [0.5 , 0.9])
    cnn_milstone_list = [i * 20 // 4 for i in range(1, 4)]
    cnn_shceduler_fe = torch.optim.lr_scheduler.MultiStepLR(cnn_optimizer_fe, milestones=cnn_milstone_list, gamma=0.1)
    cnn_shceduler_nf = torch.optim.lr_scheduler.MultiStepLR(cnn_optimizer_nf, milestones=cnn_milstone_list, gamma=0.1)
    blong_loss_spec = torch.nn.MSELoss(reduction='mean')
    blong_loss_chan = torch.nn.BCEWithLogitsLoss()
    cnn_loss_spec = torch.nn.MSELoss(reduction="mean")


    # if extf >= config.stft_window_size:
    #     train_loss_scale = extf / 2
    # else:
    #     train_loss_scale = 1
    # train_loss_scale = 1
    train_loss_scale = 8
    print(train_loss_scale)

    step = 0
    final_models_counter = 0
    for epoch in range(max(blong_epoch, cnn_epoch)):
        l = lamb * get_lambda(epoch, blong_epoch)
        blong_total_item_loss = 0
        cnn_total_item_loss = 0
        start_step = step
        for idx, ((noise_sym, sym, label, _, _), (target_noise_sym, _, _, _, _)) in enumerate(zip(train_data_loader, test_data_loader)):
            sym = sym.to(device)
            target_noise_sym = target_noise_sym.to(device)
            noise_sym = noise_sym.to(device)
            label = label.to(device)
            mixed_sym = torch.cat((noise_sym, target_noise_sym), dim=0)
            # 0 for source domain, 1 for target domain
            domain_label = torch.zeros((mixed_sym.shape[0], 1), dtype=torch.float32).to(device)
            domain_label[noise_sym.shape[0]: ] = 1
            if epoch < blong_epoch:
                # BLong_nn trainning
                blong_features = blong_fe(mixed_sym)
                blong_domain_label = blong_cd(blong_features.detach())  # detach since we do not update FE yet
                # Frist, train the channel discrimnator
                blong_loss_cd = blong_loss_chan(blong_domain_label, domain_label)
                blong_cd.zero_grad()
                blong_loss_cd.backward()
                blong_optimizer_cd.step()
                # Then we train the feature extractor and noise filter
                blong_denoised_mask = blong_nf(blong_features[:noise_sym.shape[0]])
                blong_denoised_sym = blong_denoised_mask * noise_sym
                blong_domain_label= blong_cd(blong_features)  # dot not detach since we have to update the FE now
                blong_loss_nf = blong_loss_spec(blong_denoised_sym, sym) / train_loss_scale
                blong_loss_cd = blong_loss_chan(blong_domain_label, domain_label)
                blong_total_loss =  blong_loss_nf - l * blong_loss_cd
                blong_fe.zero_grad()
                blong_nf.zero_grad()
                blong_cd.zero_grad()
                blong_total_loss.backward()
                blong_optimizer_nf.step()
                blong_optimizer_fe.step()
                blong_total_item_loss += blong_loss_nf.item()

            # CNN trainning
            if epoch < cnn_epoch:
                cnn_features = cnn_fe(noise_sym)
                cnn_denoised_mask = cnn_nf(cnn_features)
                cnn_denoised_sym = cnn_denoised_mask * noise_sym
                cnn_loss_nf = cnn_loss_spec(cnn_denoised_sym, sym) / train_loss_scale
                cnn_fe.zero_grad()
                cnn_nf.zero_grad()
                cnn_loss_nf.backward()
                cnn_optimizer_nf.step()
                cnn_optimizer_fe.step()
                cnn_total_item_loss += cnn_loss_nf.item()

            step += 1
            # outputs 
            if step % 100 == 0:
                s = "epoch:{}, step:{}".format(epoch, step)
                s1 = "blong_nn->NF_loss:{:.4f}, CD_loss:{:.4f}".format(blong_loss_nf.item(), blong_loss_cd.item())
                s2 = "cnn->NF_loss:{:.4f}".format(cnn_loss_nf.item())
                print(s, s1, s2)
                train_log.write(s + " " + s1 + " " + s2 + "\n")
                if step > 2000 and step % 1000 == 0:
                    # store the models in the final epoch
                    if epoch >= cnn_epoch-2 and epoch < cnn_epoch:
                        d2 = {"fe_state": cnn_fe.state_dict(), "nf_state": cnn_nf.state_dict()}
                        torch.save(d2, "./models/cnn_extf=" + str(extf) + "_round=" + str(round) + "_final_" + str(final_models_counter) + ".pt")
                    if epoch >= blong_epoch-2:
                        d = {"fe_state": blong_fe.state_dict(), "nf_state": blong_nf.state_dict()}
                        torch.save(d, "./models/blong_nn_extf=" + str(extf) + "_round=" + str(round) + "_final_" + str(final_models_counter) + ".pt")
        # update BLong NN learning rate
        blong_scheduler_cd.step()
        blong_scheduler_nf.step()
        blong_scheduler_fe.step()
        # update CNN learning rate
        cnn_shceduler_nf.step()
        cnn_shceduler_fe.step()
        s = "epoch={}->{}, {}".format(epoch, blong_total_item_loss / (step-start_step), cnn_total_item_loss / (step-start_step))
        print(s)
        train_log.write(s + "\n")
    d = {"fe_state": blong_fe.state_dict(), "nf_state": blong_nf.state_dict()}
    torch.save(d, "./models/blong_nn_extf=" + str(extf) + "_round=" + str(round) + "_final.pt")
    d2 = {"fe_state": cnn_fe.state_dict(), "nf_state": cnn_nf.state_dict()}
    torch.save(d2, "./models/cnn_extf=" + str(extf) + "_round=" + str(round) + "_final.pt")
    train_log.flush()
    train_log.close()


round_num = 5
if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    for extf in [127]:
        train_sym_dataset = ble_data_loader.ble_torch_dataset(extf, True, [-30, 10])
        train_data_loader = DataLoader(train_sym_dataset, batch_size=config.batch_size, shuffle=True, num_workers=6)
        test_sym_dataset = ble_data_loader.ble_torch_dataset(extf, False, [-30, 10])
        test_data_loader = DataLoader(test_sym_dataset, batch_size=config.batch_size, shuffle=True, num_workers=6)
        for round in range(round_num):
            train_toghter(train_data_loader, test_data_loader, extf, round, 20, 20)
            blong_eval(extf, round)
            
        eval_all(extf)
        # if extf >= 64:
        #     train_toghter(train_data_loader, test_data_loader, extf, 40)
        # else:
        #     train_toghter(train_data_loader, test_data_loader, extf, 30)
        # blong_nn_train(train_data_loader, test_data_loader, extf)
        # cnn_train(train_data_loader, test_data_loader, extf)
        


if f is not None:
    f.close()