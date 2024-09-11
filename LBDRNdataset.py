import os
import torch
import numpy as np
from osgeo import gdal
from torch.utils.data import Dataset
from constants import *


gdal.UseExceptions()


def merge_tiles(input_dir, output_file, split_ratio, width, height):
    tile_width = width // split_ratio
    tile_height = height // split_ratio

    tile0_file = os.path.join(input_dir, f"tile_0_0_recon.tif")
    dataset = gdal.Open(tile0_file)

    driver = gdal.GetDriverByName('GTiff')
    output_dataset = driver.Create(output_file, width, height, 
                                   dataset.RasterCount, dataset.GetRasterBand(1).DataType) 
    if output_dataset is None:
        raise ValueError(f"Failed to create {output_file}.")

    for i in range(split_ratio):
        for j in range(split_ratio):
            tile_file = os.path.join(input_dir, f"tile_{i}_{j}_recon.tif")
            tile_dataset = gdal.Open(tile_file)
            if tile_dataset is None:
                raise ValueError(f"Failed to open {tile_file}.")

            x_offset = j * tile_width
            y_offset = i * tile_height
            # Adjust tile width and height for last tiles
            tile_w = tile_width if j + 1 < split_ratio else width - x_offset
            tile_h = tile_height if i + 1 < split_ratio else height - y_offset

            data = tile_dataset.ReadAsArray()
            output_dataset.WriteArray(data, x_offset, y_offset)
            print(f"Tile {i}_{j} merged, shape: {tile_w}x{tile_h}")

    output_dataset.FlushCache()
    output_dataset = None


def split_image(input_file, output_dir, split_ratio):
    input_dataset = gdal.Open(input_file)
    if input_dataset is None:
        raise ValueError(f"Failed to open {input_file}.")

    width = input_dataset.RasterXSize
    height = input_dataset.RasterYSize
    tile_w, tile_h = width // split_ratio, height // split_ratio

    for i in range(split_ratio):
        for j in range(split_ratio):
            x_offset = j * tile_w
            y_offset = i * tile_h
            output_file = os.path.join(output_dir, f"tile_{i}_{j}.tif")
            tile_ws = tile_w
            tile_hs = tile_h
            if j + 1 == split_ratio: tile_ws = width - tile_w * j
            if i + 1 == split_ratio: tile_hs = height - tile_h * i
            gdal.Translate(output_file, input_dataset, 
                           srcWin=[x_offset, y_offset, tile_ws, tile_hs])
            print(f"Tile {i}_{j} created, shape: {tile_ws}x{tile_hs}")

    input_dataset = None


def write_tiff_with_gdal(output_path: str, array: np.array):
    # The numpy array must be data with shape CHW
    gdal_type_mapping = { # Can add more if needed
        np.uint8: gdal.GDT_Byte,
        np.uint16: gdal.GDT_UInt16,
        np.float32: gdal.GDT_Float32,
        np.float64: gdal.GDT_Float64
    }
    gdal_dtype = gdal_type_mapping.get(array.dtype.type)
    if gdal_dtype is None:
        raise ValueError("Unsupported data type in this function")
    driver = gdal.GetDriverByName('GTiff')
    # W, H, C
    dataset = driver.Create(output_path, array.shape[2], array.shape[1], array.shape[0], gdal_dtype)
    for i in range(array.shape[0]):
        band = dataset.GetRasterBand(i + 1)  
        band.WriteArray(array[i, :, :])
    dataset.FlushCache()
    dataset = None


def process(path, K, D, output_path):
    dataset = gdal.Open(path)
    img = dataset.ReadAsArray() # CHW or HW
    MSB = img >> K # CHW or HW
    LSB = img - (MSB << K) #
    LSB = LSB.astype(np.float32) / (2 ** K - 1) #
    MSB = MSB.reshape((-1, MSB.shape[-2], MSB.shape[-1])) # CHW
    LSB = LSB.reshape((-1, LSB.shape[-2], LSB.shape[-1])) # CHW
    MSB = MSB.astype(np.uint16) if MSB.max() > 255 else MSB.astype(np.uint8)
    write_tiff_with_gdal(output_path, MSB)
    C, H, W = MSB.shape
    
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
        MSB = np.pad(MSB.astype(np.float32) / MSB.max(), 
                        ((0, 0), (D, D), (D, D)),
                        mode='reflect'
                        ).transpose(1, 2, 0) # (H+2D)(W+2D)C
        colors = np.lib.stride_tricks.sliding_window_view(MSB, (2 * D + 1, 2 * D + 1), 
                                                          axis=(0, 1))
        if RELATIVE and D > 0:
            centers = MSB[D:H+D, D:W+D, :][:, :, :, np.newaxis, np.newaxis]
            colors = colors - centers
        features[:, :, num_coords:] = colors.reshape((H, W, -1))
    features = features.reshape(H * W, feature_dim)
    labels = LSB.transpose(1, 2, 0).reshape(H * W, C)
    
    return features, labels


class LBDRNDataset(Dataset):
    def __init__(self, args):
        filename = os.path.splitext(os.path.basename(args.path))[0]
        output_path = f'{args.output_dir}/{filename}_base.tif'
        features, labels = process(args.path, args.K, args.D, output_path)
        self.features = torch.from_numpy(features).to(torch.float32)
        self.labels = torch.from_numpy(labels).to(torch.float32)
        self.n_pixels = len(self.features)
        self.n_feature = self.features.shape[-1]
        self.channels = self.labels.shape[-1]
        self.n_subpixels = self.n_pixels * self.channels # HWC

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):  
        feature = self.features[idx]
        label = self.labels[idx]
        
        return feature, label
