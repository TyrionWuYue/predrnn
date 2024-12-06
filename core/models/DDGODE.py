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
        self.node_num = (self.width // 4) ** 2
        self.MSE_criterion = nn.MSELoss()
        self.MAE_criterion = nn.L1Loss()

        self.init_conv = Stem(self.frame_channel, num_hidden[0]) # Encoder
        self.decoder = Decoder(num_hidden[-1]*2, self.frame_channel+3)
        for i in range(num_layers):
            cell_list.append(
                DyGCRNCell(node_num=self.node_num, dim_in=num_hidden[i-1], hidden_dim=num_hidden[i], mem_num=32, embed_dim=10, width=self.width// 4)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.odeblock = ODEBlock(num_hidden[0], 128, self.configs.total_length - self.configs.input_length)
        self.last_conv = nn.Conv2d(num_hidden[-1], num_hidden[0], 1, 1, 0, bias=False)
    

    def get_flowmaps(self, sol_out, first_prev_embed):
        b, T, c, h, w = sol_out.size()
        pred_flows = list()
        prev = first_prev_embed.clone()

        for t in range(T):
            curr_and_prev = torch.cat([sol_out[:, t, ...], prev], dim=1)
            pred_flow = self.decoder(curr_and_prev).unsqueeze(1)
            pred_flows += [pred_flow]
            prev = sol_out[:, t, ...].clone()
        
        return pred_flows


    def get_warped_images(self, pred_flows, start_image, grid):
        b, T, c, h, w = pred_flows.shape
        pred_x = list()
        last_frame = start_image

        for t in range(T):
            pred_flow = pred_flows[:, t, ...]
            pred_flow = torch.cat([pred_flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), pred_flow[:, 1:2, :, :] /((h - 1.0) / 2.0)], dim=1)
            pred_flow = pred_flow.permute(0, 2, 3, 1)
            flow_grid = grid.clone() + pred_flow.clone()
            warped_x = nn.functional.grid_sample(last_frame, flow_grid, padding_mode="border", align_corners=False)
            pred_x += [warped_x.unsqueeze(1)]
            last_frame = warped_x.clone()

        return pred_x


    def forward(self, frames_tensor, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]
        assert self.node_num == (width // 4) * (height // 4)

        sol_y = [] # next frames
        h_t = []

        frames = frames.view(-1, self.frame_channel, height, width)
        frames_down = self.init_conv(frames)
        frames = frames.view(batch, -1, self.frame_channel, height, width)
        frames_down = frames_down.view(batch, -1, self.num_hidden[0], height// 4, width// 4)
        skip_conn_embed = frames_down[:, 0, ...].reshape(batch, -1, height//4, width//4)
        mask_true = mask_true[:, :, :1, :1, :1].repeat(1, 1, self.num_hidden[0], height// 4, width// 4)

        last_down_frame = frames_down[:, self.configs.input_length - 1, ...].reshape(batch, -1, height// 4, width// 4)
        future_down_frames = self.odeblock(last_down_frame)
        ode_down_frames = future_down_frames.view(batch, -1, self.num_hidden[0], height// 4, width// 4)
        assert ode_down_frames.shape[1] == self.configs.total_length - self.configs.input_length

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height// 4, width// 4]).to(self.configs.device)
            h_t.append(zeros)
        
        for t in range(self.configs.total_length - 1):
            # reverse schedule sampling
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames_down[:, t]
                else:
                    if t < self.configs.input_length - 2:
                        net = frames_down[:, t]
                    else:
                        net = mask_true[:, t - 1] * frames_down[:, t] + (1 - mask_true[:, t - 1]) * ode_down_frames[:, t - self.configs.input_length + 1]


            h_t[0] = self.cell_list[0](net, h_t[0])
            for i in range(1, self.num_layers):
                h_t[i] = self.cell_list[i](h_t[i-1], h_t[i])
            gen_frame = self.last_conv(h_t[-1])
            sol_y.append(gen_frame)
        
        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        sol_y = torch.stack(sol_y, dim=0).transpose(0,1).contiguous()
        pred_outputs = self.get_flowmaps(sol_out=sol_y, first_prev_embed=skip_conn_embed)
        pred_outputs = torch.cat(pred_outputs, dim=1)
        pred_flows, pred_intermediates, pred_masks = \
            pred_outputs[:, :, :2, ...], pred_outputs[:, :, 2:2+self.frame_channel, ...], torch.sigmoid(pred_outputs[:, :, 2+self.frame_channel:, ...])

        grid_x = torch.linspace(-1.0, 1.0, width).view(1, 1, width, 1).expand(batch, height, -1, -1)
        grid_y = torch.linspace(-1.0, 1.0, height).view(1, height, 1, 1).expand(batch, -1, width, -1)
        grid = torch.cat([grid_x, grid_y], 3).float().to(frames_tensor.device)

        # Wrapping
        last_frame = frames[:, 0, ...]
        warped_pred_x = self.get_warped_images(pred_flows=pred_flows, start_image=last_frame, grid=grid)
        warped_pred_x = torch.cat(warped_pred_x, dim=1)
        warped_pred_x = warped_pred_x.reshape(-1, self.frame_channel, self.configs.img_width, self.configs.img_width)
        warped_pred_x = warped_pred_x.reshape(batch, -1, self.frame_channel, height, width)
        
        pred_x = pred_masks*warped_pred_x + (1-pred_masks)*pred_intermediates
        next_frames = pred_x.view(batch, -1, self.frame_channel, height, width)
        next_frames = next_frames.permute(0, 1, 3, 4, 2).contiguous()
        # inter loss
        frame_diff = frames_tensor[:, 1:, ...] - frames_tensor[:, :-1, ...]
        pred_intermediates = pred_intermediates.permute(0, 1, 3, 4, 2)
        inetr_loss = self.MSE_criterion(frame_diff, pred_intermediates)

        mse_loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])

        ode_loss = self.MSE_criterion(ode_down_frames, frames_down[:, -(self.configs.total_length - self.configs.input_length):])
        loss  = mse_loss + inetr_loss + 0.03*ode_loss

        return next_frames, loss
