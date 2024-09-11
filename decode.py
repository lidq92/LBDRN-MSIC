import os
import sys
import math
import time
import random
import argparse
import subprocess
import tracemalloc

import fpzip
import torch
import numpy as np
from osgeo import gdal

import logger
from constants import *
from LBDRNmodel import LBDRNModel
from LBDRNdataset import merge_tiles, write_tiff_with_gdal


gdal.UseExceptions()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_image_header(bitstream):
    ptr = 0
    n_bytes_header = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1
    split_ratio = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1
    width = int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
    ptr += 2
    height = int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
    ptr += 2
    KD = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1
    K = KD >> 4
    D = KD & 0x0F
    bcnl = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1
    bc = 2 ** (bcnl >> 4)
    nl = bcnl & 0x0F
    nn_bytes_list, base_bytes_list = [], []
    for _ in range(split_ratio ** 2):
        nn_bytes = int.from_bytes(bitstream[ptr: ptr + 3], byteorder='big', signed=False)
        ptr += 3
        nn_bytes_list.append(nn_bytes)
    for _ in range(split_ratio ** 2):
        base_bytes = int.from_bytes(bitstream[ptr: ptr + 4], byteorder='big', signed=False)
        ptr += 4
        base_bytes_list.append(base_bytes)

    return n_bytes_header, split_ratio, width, height, K, bc, nl, D, nn_bytes_list, base_bytes_list
    
        
def test(bitstream, dirname, filename, nn_bytes, base_bytes):

    sub_nn_bitstream = bitstream[:nn_bytes]
    sub_nn_bitstream_path = f'{dirname}/{filename}_nn.bin'
    with open(sub_nn_bitstream_path, 'wb') as f_out: f_out.write(sub_nn_bitstream)
    bitstream = bitstream[nn_bytes:]

    recon_path = f'{dirname}/{filename}_recon.tif'
    jp2_path = f'{dirname}/{filename}_base.jp2'
    sub_base_bitstream = bitstream[:base_bytes]
    with open(jp2_path, 'wb') as f_out: f_out.write(sub_base_bitstream)
    bitstream = bitstream[base_bytes:]

    cmd_decode = f"gdal_translate -of GTiff {jp2_path} {recon_path}"
    r = sh(cmd_decode) 
    logger.log.info(r)
    
    dataset = gdal.Open(recon_path)
    base = dataset.ReadAsArray().astype(np.uint16) # CHW or HW uint16!!!
    base = base.reshape((-1, base.shape[-2], base.shape[-1])) # CHW
    C, H, W = base.shape
    num_colors = C * (2 * D + 1) ** 2 * USE_COLORS
    num_coords =  (2 * N_FREQ * EMBEDDING + 1) * 2 * USE_COORDINATES
    feature_dim = num_coords + num_colors
    features = np.zeros((H, W, feature_dim), dtype=np.float32) # 
    if USE_COORDINATES:
        coords_h, coords_w = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')    
        ph = 2 * coords_h / (H - 1) - 1
        pw = 2 * coords_w / (W - 1) - 1
        coords = np.stack([ph, pw], axis=-1).astype(np.float32)
        if EMBEDDING:
            sin_part = np.sin(SIGMA ** np.arange(N_FREQ) * np.pi * coords[..., np.newaxis])
            cos_part = np.cos(SIGMA ** np.arange(N_FREQ) * np.pi * coords[..., np.newaxis])
            coords = np.concatenate([coords[..., np.newaxis], sin_part, cos_part], axis=-1)
        coords = coords.reshape((H, W, -1))
        features[:, :, :num_coords] = coords.reshape(H, W, -1)
    if USE_COLORS:
        base_pad = np.pad(base.astype(np.float32) / base.max(), 
                        ((0, 0), (D, D), (D, D)),
                        mode='reflect'
                        ).transpose(1, 2, 0) # (H+2D)(W+2D)C
        colors = np.lib.stride_tricks.sliding_window_view(base_pad, (2 * D + 1, 2 * D + 1), axis=(0, 1))
        if RELATIVE and D > 0:
            centers = base_pad[D:H+D, D:W+D, :][:, :, :, np.newaxis, np.newaxis]
            colors = colors - centers
        features[:, :, num_coords:] = colors.reshape((H, W, -1))
    features = features.reshape(H * W, feature_dim)

    model = LBDRNModel(dim_in=features.shape[-1], 
                          dim_hidden=bc, 
                          dim_out=C,
                          num_layers=nl,
                          # activation=torch.nn.ReLU() # Default: Sine
                          )
    model = model.to(DEVICE)
    
    with open(sub_nn_bitstream_path,'rb') as f: compressed_bytes = f.read()
    params = fpzip.decompress(compressed_bytes, order='C')[0][0][0]
    k = 0
    state_dict = {}
    for param_tensor in model.state_dict():
        values = params[k:k+model.state_dict()[param_tensor].numel()].reshape(model.state_dict()[param_tensor].size())
        state_dict[param_tensor] = torch.from_numpy(values)
        k = k + model.state_dict()[param_tensor].numel()
    model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(features).to(torch.float32)
        y_pred = torch.zeros(x.shape[0], C).to(DEVICE) # Save CUDA Memory? to('cpu')
        # y_pred = model(x.to(DEVICE))
        bs = 2 ** 22 # Avoid CUDA Out of Memory
        for b in range(math.ceil(x.shape[0] / bs)):
            xb = x[bs*b:bs*(b+1)].to(DEVICE)
            y_pred[bs*b:bs*(b+1)] = model(xb) # Save CUDA Memory? to('cpu')
        residual = torch.round(y_pred * (2 ** K -1 )).to('cpu').numpy() # 
        residual = residual.reshape(H, W, C)
        residual = np.transpose(residual, axes=(2, 0, 1))
        image = np.round((base << K).astype(np.float32) + residual).astype(np.uint16)
        write_tiff_with_gdal(recon_path, image)
        logger.log.info(f'Recon: {recon_path}')
        subprocess.call(f'rm -f {jp2_path}', shell=True)
        subprocess.call(f'rm -f {jp2_path}.aux.xml', shell=True)
        subprocess.call(f'rm -f {sub_nn_bitstream_path}', shell=True)
    
    return bitstream


def sh(cmd, input=''): 
    rst = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE, input=input.encode('utf-8'))
    assert rst.returncode == 0, rst.stderr.decode('utf-8')
    return rst.stdout.decode('utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LBDRN-RSIC')
    parser.add_argument('--seed', type=int, default=19920517)
    parser.add_argument('-i', '--bin_path', type=str, help='binstream path')
    parser.add_argument('-org', '--org_path', type=str, default=None, help='org path')
    args = parser.parse_args()
    torch.manual_seed(args.seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.utils.backcompat.broadcast_warning.enabled = True

    # tracemalloc.start()

    dirname, basename = os.path.split(args.bin_path)
    filename = os.path.splitext(basename)[0]
    if os.path.exists(f'{dirname}/decode.txt'):
        decoded = False
        with open(f'{dirname}/decode.txt', 'r') as file:
            content = file.read()
            if "bpsp" in content:
                decoded = True
                print('Bitstream already decoded!')
        if decoded:
            sys.exit()
    logger.create_logger(dirname, 'decode.txt')
    logger.log.info(f'Binstream: {args.bin_path}')
    start_time = time.time()
    with open(args.bin_path, 'rb') as fin: bitstream = fin.read()

    n_bytes_header, split_ratio, width, height, K, bc, nl, D, nn_bytes_list, base_bytes_list = read_image_header(bitstream)
    bitstream = bitstream[n_bytes_header:]
    
    bin_path = args.bin_path
    recon_path = f'{dirname}/{basename[:-4]}_recon.tif'
    if split_ratio > 1:
        for i in range(split_ratio):
            for j in range(split_ratio):
                bitstream = test(bitstream, dirname, filename=f'tile_{i}_{j}',
                                 nn_bytes=nn_bytes_list[i*split_ratio+j],
                                 base_bytes=base_bytes_list[i*split_ratio+j])

        merge_tiles(dirname, recon_path, split_ratio, width, height)
        for i in range(split_ratio):
            for j in range(split_ratio):
                subprocess.call(f'rm -f {dirname}/tile_{i}_{j}_recon.tif', shell=True)

    else:
        bitstream = test(bitstream,dirname, filename, nn_bytes_list[0], base_bytes_list[0])

    end_time = time.time()
    logger.log.info(f'Time elapsed: {end_time - start_time}')

    # current, peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # logger.log.info(f"Current memory usage: {current / 10**6:.2f} MB")
    # logger.log.info(f"Peak memory usage: {peak / 10**6:.2f} MB")

    if args.org_path is not None:
        dataset = gdal.Open(args.org_path)
        org_img = dataset.ReadAsArray() # CHW
        dataset = gdal.Open(recon_path)
        rec_img = dataset.ReadAsArray() # CHW
        bytes = os.path.getsize(bin_path)
        mse_value = np.mean((org_img.astype(np.float32) - rec_img.astype(np.float32)) ** 2) #
        logger.log.info(f"MSE: {mse_value}")
        peak = 10000 # np.max(org_img) # 
        psnr = 10 * np.log10(peak ** 2 / mse_value)
        logger.log.info(f"PSNR: {psnr}")   
        n_subpixels = np.prod(org_img.shape)
        logger.log.info(f"Total size: {bytes} bytes, bpsp={bytes * 8 / n_subpixels}")
        if True: # False: # Delete the reconstructed image?
            subprocess.call(f'rm -f {recon_path}', shell=True)
    