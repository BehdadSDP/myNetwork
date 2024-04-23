import torch
import torch.nn as nn

loss_func =  nn.MSELoss()

def dehazeloss(clear_image, rst_output):
    loss_dehaze = loss_func(clear_image, rst_output)
    loss_dehaze = loss_dehaze.unsqueeze(0)
    return loss_dehaze 