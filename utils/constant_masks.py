import os
from pathlib import Path
import numpy as np
import torch


def load_constant_masks(npy_path='./utils', shape=(721, 1440), ds_rate=1, device='cpu'):
    # print(npy_path)
    npy_path = Path(npy_path) / 'constant_masks.npy'
    if not os.path.exists(npy_path):
        raise Exception(
            'Constant masks .npy not exist! Run convert_constant_masks_to_numpy in constant_masks.py first!')
    h_range = int(shape[0] // ds_rate * ds_rate)
    w_range = int(shape[1] // ds_rate * ds_rate)
    land_mask, soil_type, topography = [
        torch.tensor(arr[:h_range:ds_rate, :w_range:ds_rate], dtype=torch.float32, device=device)
        for arr in np.load(npy_path)]
    for mask in [land_mask, soil_type, topography]:
        h_ds, w_ds = mask.shape
        mask = mask.reshape(-1)
        mean_mask = mask.mean(-1, keepdim=True).detach()  # number
        mask -= mean_mask
        stdev_mask = torch.sqrt(torch.var(mask, dim=-1, keepdim=True, unbiased=False) + 1e-5)  # number
        mask /= stdev_mask
        mask = mask.reshape(h_ds, w_ds)
    return land_mask, soil_type, topography


if __name__ == '__main__':
    # convert_constant_masks_to_numpy()
    land_mask, soil_type, topography = load_constant_masks()
    print(topography[247, 347])  # Qomolangma
    print(topography[315, 569])  # Mariana

    import matplotlib.pyplot as plt
    h, w = 721, 1440
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=200)
    plt.xticks(np.linspace(0, w, 7), ['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°'])
    plt.xlabel('longitude')
    plt.yticks(np.linspace(0, h, 7), ['90°N', '60°N', '30°N', '0°', '30°S', '60°S', '90°S'])
    plt.ylabel('latitude')
    plt.grid(True)
    plt.title('Global Land-sea mask')
    plt.imshow(land_mask, cmap='jet')
    plt.colorbar(shrink=1 / 1.25)
    plt.savefig('../img/land_sea_mask.png')
    # plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(w / 100, h / 100), dpi=200)
    plt.xticks(np.linspace(0, w, 7), ['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°'])
    plt.xlabel('longitude')
    plt.yticks(np.linspace(0, h, 7), ['90°N', '60°N', '30°N', '0°', '30°S', '60°S', '90°S'])
    plt.ylabel('latitude')
    plt.grid(True)
    plt.title('Global Soil type mask')
    plt.imshow(soil_type, cmap='jet')
    plt.colorbar(shrink=1 / 1.25)
    plt.savefig('../img/soil_type_mask.png')
    # plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(w / 100, h / 100), dpi=200)
    plt.xticks(np.linspace(0, w, 7), ['0°', '60°E', '120°E', '180°', '120°W', '60°W', '0°'])
    plt.xlabel('longitude')
    plt.yticks(np.linspace(0, h, 7), ['90°N', '60°N', '30°N', '0°', '30°S', '60°S', '90°S'])
    plt.ylabel('latitude')
    plt.grid(True)
    plt.title('Global Topography mask')
    plt.imshow(topography, cmap='gnuplot2')
    plt.colorbar(shrink=1 / 1.25)
    plt.savefig('../img/topography_mask.png')
    # plt.show()
    plt.close(fig)
