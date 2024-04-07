import torch
import torch.nn as nn

loss_func =  nn.MSELoss()

def dehazeloss(clear_image, rst_output):
    loss_dehaze = loss_func(clear_image, rst_output)
    l_dehaze = torch.unsqueeze(loss_dehaze, dim = 0)
    return l_dehaze 