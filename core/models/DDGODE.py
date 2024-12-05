import torch
import torch.nn as nn
from core.layers.DyGCNCell import DyGCRNCell
from core.layers.ODE import ODEBlock
from core.layers.ConvLayers import *


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        self.width = configs.img_width // configs.patch_size
        self.node_num = (self.width // 2) ** 2
        self.MSE_criterion = nn.MSELoss()

        self.init_conv = DoubleConv(self.frame_channel, num_hidden[0])
        self.down = DownSample(in_channels=num_hidden[0], out_channels=num_hidden[0])
        self.up = UpSample(in_channels=num_hidden[num_layers - 1]*2, out_channels=num_hidden[num_layers - 1], h=self.width//2, w=self.width//2)
        for i in range(num_layers):
            cell_list.append(
                DyGCRNCell(node_num=self.node_num, dim_in=num_hidden[i-1], hidden_dim=num_hidden[i], cheb_k=2, mem_num=32, embed_dim=10)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.odeblock = ODEBlock(num_hidden[0], 128, self.configs.total_length - 1)
        self.conv_h2z = nn.Conv2d(num_hidden[num_layers-1], num_hidden[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_last = nn.Conv2d(num_hidden[0], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)
    
    def forward(self, frames_tensor, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]
        assert self.node_num == (width // 2) * (height // 2)

        next_frames = []
        h_t = []

        frames = frames.view(-1, self.frame_channel, height, width)
        frames = self.init_conv(frames)
        frames_down = self.down(frames) # DwonSample
        frames_down = frames_down.view(batch, -1, self.num_hidden[0], height//2, width//2)
        frames = frames.view(batch, -1, self.num_hidden[0], height, width)

        first_frame = frames[:, 0, ...].reshape(batch, -1, self.num_hidden[0])
        future_frames = self.odeblock(first_frame)
        future_frames = future_frames.view(batch, -1, self.num_hidden[0], height, width)
        assert future_frames.shape[1] == self.configs.total_length - 1

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height//2, width//2]).to(self.configs.device)
            h_t.append(zeros)
        
        for t in range(self.configs.total_length - 1):
            # reverse schedule sampling
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames_down[:, t]
                else:
                    net = mask_true[:, t - 1] * frames_down[:, t] + (1 - mask_true[:, t - 1]) * z_gen
            else:
                if t < self.configs.input_length:
                    net = frames_down[:, t]
                else:
                    net = z_gen

            for i in range(self.num_layers):
                h_t[i] = self.cell_list[i](net, h_t[i])
            z_gen = self.conv_h2z(h_t[-1])
            frames_up = self.up(z_gen, future_frames[:, t])
            x_gen = self.conv_last(frames_up)
            next_frames.append(x_gen)
        
        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        z_loss = self.MSE_criterion(future_frames, frames[:, 1:])
        loss += z_loss
        return next_frames, loss
            
        
        