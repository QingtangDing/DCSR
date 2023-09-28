import torch
from option import args
from trainer import Trainer
import utility
import data
import model
import loss
import numpy as np

if __name__ == '__main__':
    mse_loss = torch.nn.MSELoss(reduction="sum")
    epochs = torch.linspace(0, 1000, 21)
    weights_error_with_textures = []
    weights_error_middle_textures = []
    weights_error_without_textures = []
    for epoch in epochs:
        weights_path_full_data = './weights_data_full_'+str(int(epoch))+'.pt'
        weights_path_data_with_texture = './weights_data_with_texture_'+str(int(epoch))+'.pt'
        weights_path_data_without_texture = './weights_data_without_texture_'+str(int(epoch))+'.pt'
        weights_path_data_middle_texture = './weights_data_complex_texture_' + str(int(epoch)) + '.pt'
        weights_path_init = "./weights_init_"+str(int(epoch))+'.pt'
        weights_init = torch.load(weights_path_init).unsqueeze(dim=0)
        weights_full_data = torch.load(weights_path_full_data).unsqueeze(dim=0)
        weights_data_with_texture = torch.load(weights_path_data_with_texture).unsqueeze(dim=0)
        weights_data_without_texture = torch.load(weights_path_data_without_texture).unsqueeze(dim=0)
        weights_data_middle_texture = torch.load(weights_path_data_middle_texture).unsqueeze(dim=0)
        ts_full_data - weights_init) / mse_loss(weights_full_data, weights_init))
        weights_error_with_texture = mse_loss(weights_data_with_texture, weights_full_data)/mse_loss(weights_data_with_texture, weights_init)
        weights_error_with_textures.append(weights_error_with_texture)
        weights_error_middle_texture = mse_loss(weights_data_middle_texture, weights_full_data) / mse_loss(
            weights_data_middle_texture, weights_init)
        weights_error_middle_textures.append(weights_error_middle_texture)
        weights_error_without_texture = mse_loss(weights_data_without_texture, weights_full_data) / mse_loss(weights_data_without_texture, weights_init)
        weights_error_without_textures.append(weights_error_without_texture)
    np.save('./weights_error_with_textures.npy', weights_error_with_textures)
    np.save('./weights_error_complex_textures.npy', weights_error_middle_textures)
    np.save('./weights_error_without_textures.npy', weights_error_without_textures)