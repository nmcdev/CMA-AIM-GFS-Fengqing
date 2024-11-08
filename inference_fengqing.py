import os
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timedelta
import calendar
import json
import re
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import torch
import onnx
import onnxruntime as rt
from utils.stats import get_stats
from utils.timefeatures import time_features
from utils.constant_masks import load_constant_masks
import concurrent.futures
from einops import rearrange
import argparse 
import copy

# Function to handle command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Daily inference for weather forecasting")
    parser.add_argument('--dataset_path', type=str, required=True, help="Root directory for input data")
    parser.add_argument('--datetime', type=str, required=True, help="Datetime for the forecast in YYYYMMDDHH format")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory for output GRIB2 files")
    return parser.parse_args()

options = rt.SessionOptions()
options.enable_cpu_mem_arena = False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
options.intra_op_num_threads = 16
cuda_provider_options = {
    "arena_extend_strategy": "kSameAsRequested",
    "cudnn_conv_algo_search": "HEURISTIC",
    "has_user_compute_stream": True,
    "gpu_mem_limit": 1024 * 1024 * 1024 * 40,
}

def normalize(denorm_upper, upper_mean, upper_std, denorm_surface, surface_mean, surface_std):
    # denorm_upper[0, 5] = denorm_upper[0, 65]
    # denorm_upper = denorm_upper[:, :65]
    upper = (denorm_upper - upper_mean[:, 0]) / upper_std[:, 0]
    surface = (denorm_surface - surface_mean[:, 0]) / surface_std[:, 0]
    return upper, surface

def interpolate_data(data):
    last_row = data[-1, :]
    data_extended = np.vstack([data, last_row])
    return data_extended

def get_data(dataset_path, year, month, day, hour, cfg, pre=False):
    air_path = Path(dataset_path) / f'{year}' / f'{month:02d}{day:02d}' / f'pressure_{hour:02d}0000.npy'
    surf_path = Path(dataset_path) / f'{year}' / f'{month:02d}{day:02d}' / f'surface_{hour:02d}0000.npy'
    air_data = np.load(air_path).astype(np.float32) # b 1 lvl*v_u h w
    surf_data = np.load(surf_path).astype(np.float32) # b 1 v_s h w
    surf_data = surf_data[:, :, :len(cfg['surf_vars'])] # for precipitation
    return air_data.reshape(cfg['batch_size'], len(cfg['lvls'])*len(cfg['upper_vars']), len(cfg['lats']), len(cfg['lons'])), \
        surf_data.reshape(cfg['batch_size'], len(cfg['surf_vars']), len(cfg['lats']), len(cfg['lons']))

@torch.no_grad()
def inference(cfg, input_time, model_path, stats, spatial_condition, dataset_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    start_time = input_time
    real_input_time = input_time - timedelta(hours=cfg['lead_time'])
    ort_session = rt.InferenceSession(model_path, sess_options=options, \
        providers=[('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider'])

    print(f'Inference using models {model_path}:')
    upper_mean, upper_std, surface_mean, surface_std, res_upper_std, res_surface_std = stats

    input_upper, input_surface = [], []
    with torch.cuda.amp.autocast():
        print('loading input data...   ', end='')
        tic = time.time()
        for i in range(cfg['n_frames_in']):
            output_time = real_input_time + timedelta(hours=cfg['lead_time'] * i)
            year, month, day, hour = output_time.year, output_time.month, output_time.day, output_time.hour
            air_data, surf_data = get_data(dataset_path, year, month, day, hour, cfg)
            air_data = (air_data - upper_mean[:, 0]) / upper_std[:, 0]
            surf_data = (surf_data - surface_mean[:, 0]) / surface_std[:, 0]
            input_upper.append(air_data)
            input_surface.append(surf_data)
        toc = time.time()
        print(f'done. time: {(toc - tic):.3f}s')

        before = datetime.now()
        prev_mmdd, mmdd = None, real_input_time.strftime('%m%d')
        cfg['pred_len'] = 60
        output_upper_list = []
        output_surface_list = []
        for i in range(cfg['pred_len']):
            output_time = real_input_time + timedelta(hours=cfg['lead_time'] * (i + cfg['n_frames_in']))
            temporal_condition = time_features(pd.to_datetime(output_time), freq='h')[:, 0].reshape(1, 4).astype(np.float16)
            if i == 0 or prev_mmdd != mmdd:
                doy = output_time.timetuple().tm_yday
                if mmdd >= '0229' and calendar.isleap(output_time.year):
                    doy -= 1
                doy -= 1
            doy_mask = np.zeros([365],dtype=np.float32)
            doy_mask[doy] = 1
            
            rel_mask = np.zeros([60],dtype=np.float32)
            rel_mask[i] = 1
            input_names = [input.name for input in ort_session.get_inputs()]
            out_upper, out_surface = ort_session.run(None, \
                    {'input_upper': np.concatenate(input_upper, axis=1).astype(np.float32), 
                    'input_surface': np.concatenate(input_surface, axis=1).astype(np.float32),
                    'temporal_condition': temporal_condition, 'spatial_condition': spatial_condition,
                    'res_upper_std': res_upper_std,
                    'upper_std':upper_std,
                    'upper_mean':upper_mean,
                    'res_surface_std':res_surface_std,
                    'surface_std':surface_std,
                    'surface_mean':surface_mean,
                    'day_of_year': doy_mask,
                    'relative_position':rel_mask})
            input_upper.pop(0)
            input_surface.pop(0)
            upper, surface = normalize(copy.deepcopy(out_upper), upper_mean, upper_std, out_surface, surface_mean, surface_std)
            output_upper_list.append(out_upper[:, :65])
            output_surface_list.append(out_surface[:, :5])
            input_upper.append(upper)
            input_surface.append(surface)
            print(i, datetime.now() - before)
            before = datetime.now()
            
        year, month, day, hour = start_time.year, start_time.month, start_time.day, start_time.hour    
        year_str = str(year).rjust(2, '0')
        month_str = str(month).rjust(2, '0')
        day_str = str(day).rjust(2, '0')
        hour_str = str(hour).rjust(2, '0')
        output_upper_list = np.concatenate(output_upper_list, axis=0) # T,C,H,W
        output_surface_list = np.concatenate(output_surface_list, axis=0) # T,C,H,W
        output_directory = Path(args.output_dir) / year_str / f"{year_str}{month_str}{day_str}{hour_str}"
        os.makedirs(output_directory)
        np.save(output_directory / "FengQing_v1_upper.npy", output_upper_list)
        np.save(output_directory / "FengQing_v1_surface.npy", output_surface_list)
    
if __name__ == "__main__":
    args = parse_arguments()

    model_paths = ['./onnx/fengqing.onnx']

    with open('./utils/cfg_15days.pt', 'rb') as f:
        cfg = pickle.load(f)

    stats = get_stats()
    spatial_condition = np.stack([mask.detach().cpu().numpy() for mask in load_constant_masks()], axis=0)

    print('\n Start daily inference, current time:', datetime.now())

    # Parse the datetime argument including hours
    input_time = datetime.strptime(args.datetime, '%Y%m%d%H')

    inference(cfg, input_time, model_paths[0], stats, spatial_condition, args.dataset_path, args.output_dir)

    print(f'\nDaily inference for {input_time.strftime("%Y-%m-%d-%H")} done, current time: {datetime.now()}.\n\n')