import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
sys.path.append("./scripts/")
import config

# The equivalent kernel size of the kernel with dilation = k + (k-1) * (d-1)
stft_chan_num = 8
raw_chan_num = 8
output_chan_num = 4
disc_fc_size = 64

class STFTFeatureExtractor(nn.Module):
    def __init__(self, extf):
        super(STFTFeatureExtractor, self).__init__()
        self.extf = extf 

        self.conv = nn.Sequential(
            # cnn1
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.Conv2d(2, stft_chan_num, kernel_size=(1, 3), dilation=(1, 1)),
            nn.BatchNorm2d(stft_chan_num),
            nn.ReLU(),
            #cnn2
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Conv2d(stft_chan_num, stft_chan_num, kernel_size=(3, 1), dilation=(1, 1)),
            nn.BatchNorm2d(stft_chan_num),
            nn.ReLU(),
            # cnn3
            nn.ZeroPad2d(1),
            nn.Conv2d(stft_chan_num, stft_chan_num, kernel_size=(3, 3), dilation=(1, 1)),
            nn.BatchNorm2d(stft_chan_num),
            nn.ReLU(),
            # cnn4
            nn.ZeroPad2d((1, 1, 2, 2)),
            nn.Conv2d(stft_chan_num, stft_chan_num, kernel_size=(3, 3), dilation=(2, 1)),
            nn.BatchNorm2d(stft_chan_num),
            nn.ReLU(),
            #cnn5
            nn.ZeroPad2d((1, 1, 4, 4)),
            nn.Conv2d(stft_chan_num, stft_chan_num, kernel_size=(3, 3), dilation=(4, 1)),
            nn.BatchNorm2d(stft_chan_num),
            nn.ReLU(),
            #cnn6
            # nn.ZeroPad2d((1, 1, 4, 4)),
            nn.Conv2d(stft_chan_num, output_chan_num, kernel_size=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(output_chan_num),
            nn.ReLU(),
        )

        # lstm_size = config.stft_window_size * 2 * stft_chan_num
        # self.lstm = nn.LSTM(lstm_size, lstm_size, batch_first=True, bidirectional=True)

        # self.flat = nn.Flatten(start_dim=1, end_dim=-1)
    
    def forward(self, x):
        # x: [B, 2, H, W] H: frequency bins, W: time
        out = self.conv(x)  # [B, stft_chan_num, H, W]
        return out 
        # out = out.transpose(1, 3).contiguous()  # [B, W, H, stft_chan_num]
        # out = out.view(x.shape[0], stft_chan_num, -1).contiguous()  # [B, W, H*stft_chan_num]
        # out, _ = self.lstm(out)  # [B, W, 2*H*stft_chan_num]
        # out = F.relu(out).contiguous()

class RawFeatureExtractor(nn.Module):
    def __init__(self, extf):
        super(RawFeatureExtractor, self).__init__()
        self.extf = extf
        self.conv = nn.Sequential(
            # cnn1
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.Conv2d(4, raw_chan_num, kernel_size=(1, 3), dilation=(1, 1)),
            nn.BatchNorm2d(raw_chan_num),
            nn.ReLU(),
            #cnn2
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Conv2d(raw_chan_num, raw_chan_num, kernel_size=(3, 1), dilation=(1, 1)),
            nn.BatchNorm2d(raw_chan_num),
            nn.ReLU(),
            # cnn3
            nn.ZeroPad2d(1),
            nn.Conv2d(raw_chan_num, raw_chan_num, kernel_size=(3, 3), dilation=(1, 1)),
            nn.BatchNorm2d(raw_chan_num),
            nn.ReLU(),
            # cnn4
            nn.ZeroPad2d((1, 1, 2, 2)),
            nn.Conv2d(raw_chan_num, raw_chan_num, kernel_size=(3, 3), dilation=(2, 1)),
            nn.BatchNorm2d(raw_chan_num),
            nn.ReLU(),
            #cnn5
            nn.ZeroPad2d((1, 1, 4, 4)),
            nn.Conv2d(raw_chan_num, raw_chan_num, kernel_size=(3, 3), dilation=(4, 1)),
            nn.BatchNorm2d(raw_chan_num),
            nn.ReLU(),
            #cnn6
            # nn.ZeroPad2d((1, 1, 4, 4)),
            nn.Conv2d(raw_chan_num, output_chan_num, kernel_size=(1, 1), dilation=(1, 1)),
            nn.BatchNorm2d(output_chan_num),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: [B, 4, extf*8]
        x = x.view(x.shape[0], x.shape[1], self.extf, config.sample_pre_symbol)
        x = x.transpose(2, 3).contiguous()  # [B, 4, 8, extf]
        out = self.conv(x)
        return out

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, extf):
        super(LSTMFeatureExtractor, self).__init__()
        self.extf = extf
        lstm_size = 2 * output_chan_num * config.sample_pre_symbol
        self.lstm = nn.LSTM(lstm_size, lstm_size, batch_first=True, bidirectional=True)

    def forward(self, stft_x, raw_x):
        # stft_x: [B, output_chan_num, 8, extf]
        # raw_x: [B, output_chan_num, 8, extf]
        x = torch.cat([stft_x, raw_x], dim=1)  # [B, 2*output_chan_num, 8, extf]
        x = x.transpose(1, 3).contiguous()  # [B, extf, 8, 2*output_chan_num]
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()  # [B, extf, 8*2*output_chan_num]
        out, _ = self.lstm(x)  # [B, extf, 2*8*2*output_chan_num]
        out = F.relu(out).contiguous()
        return out

class Discriminator(nn.Module):
    def __init__(self, extf):
        super(Discriminator, self).__init__()
        self.extf = extf
        self.fc = nn.Sequential(
            nn.Linear(4 * extf * config.sample_pre_symbol * output_chan_num, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
    def forward(self, x):
        x = x.view(x.shape[0], -1).contiguous()
        out = self.fc(x)
        return out





