from pathlib import Path
import numpy as np

def get_stats(path='./mean_std'):
    upper_mean = np.load(Path(path) / 'upper_mean.npy', allow_pickle=True)[None, None, :, :, :]
    upper_std = np.load(Path(path) / 'upper_std.npy', allow_pickle=True)[None, None, :, None]
    surface_mean = np.load(Path(path) / 'surface_mean.npy', allow_pickle=True)[None, None, :, :, :]
    surface_std = np.load(Path(path) / 'surface_std.npy', allow_pickle=True)[None, None, :, None]
    res_upper_std = np.load(Path(path) / 'res_upper_std.npy', allow_pickle=True)[None, None, :, None]
    res_surface_std = np.load(Path(path) / 'res_surface_std.npy', allow_pickle=True)[None, None, :, None]
    # print(upper_mean.shape, upper_std.shape, surface_mean.shape, surface_std.shape, res_upper_std.shape, res_surface_std.shape)
    return upper_mean, upper_std, surface_mean, surface_std, res_upper_std, res_surface_std