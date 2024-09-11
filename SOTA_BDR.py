import os
import sys
import csv
import shutil
import zipfile
import subprocess
import os.path as osp
from glob import glob


import cv2
import numpy as np
from osgeo import gdal


def write_tiff_with_gdal(output_path: str, array: np.array):
    # The numpy array must be data with shape CHW
    gdal_type_mapping = {  # Can add more if needed
        np.uint8: gdal.GDT_Byte,
        np.uint16: gdal.GDT_UInt16,
        np.float32: gdal.GDT_Float32,
        np.float64: gdal.GDT_Float64
    }
    gdal_dtype = gdal_type_mapping.get(array.dtype.type)
    if gdal_dtype is None: raise ValueError("Unsupported data type in this function")
    driver = gdal.GetDriverByName('GTiff')
    # W, H, C
    dataset = driver.Create(output_path, array.shape[2], array.shape[1], array.shape[0], gdal_dtype)
    for i in range(array.shape[0]):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(array[i, :, :])
    dataset.FlushCache()


def gen_bgr_div(in_files, div_dir, div_num_h, div_num_w, with_zeros, process_extra_spectral_as_bgr):
    if osp.exists(div_dir): shutil.rmtree(div_dir)
    os.makedirs(div_dir, exist_ok=True)
    for file_path in in_files:
        print(file_path)
        target = (gdal.Open(file_path).ReadAsArray()).transpose(1, 2, 0)
        if not with_zeros:
            assert np.all(target <= 8191)
            target <<= 3
        div_h = target.shape[0] // div_num_h
        div_w = target.shape[1] // div_num_w
        div_c = 3
        div_num_c = target.shape[2] // div_c if process_extra_spectral_as_bgr else 1
        for div_idx_h in range(div_num_h):
            for div_idx_w in range(div_num_w):
                slice_end_h = (div_h * (div_idx_h + 1)) if div_idx_h != div_num_h - 1 else None
                slice_end_w = (div_w * (div_idx_w + 1)) if div_idx_w != div_num_w - 1 else None
                for div_idx_c in range(div_num_c):
                    div = target[div_h * div_idx_h: slice_end_h,
                                 div_w * div_idx_w: slice_end_w,
                                 div_c * div_idx_c: div_c * (div_idx_c + 1)]
                    print(f'Gen Div{div_idx_h}_{div_idx_w}_{div_idx_c}: {div.shape}')
                    cv2.imwrite(osp.join(div_dir, osp.split(file_path)[1] +
                                         f'_Div{div_idx_h}_{div_idx_w}_{div_idx_c}.png'), div)


def read_from_div(file_path, out_dir, in_bits, div_num_h, div_num_w, with_zeros, process_extra_spectral_as_bgr,
                  save_recon_tif=True):
    org_filename = osp.split(file_path)[1]
    target = gdal.Open(file_path).ReadAsArray().transpose(1, 2, 0)
    recon_im_path = osp.join(out_dir, org_filename + '_recon.png')
    if osp.exists(recon_im_path): return target, cv2.imread(recon_im_path, -1)
    div_h = target.shape[0] // div_num_h
    div_w = target.shape[1] // div_num_w
    div_c = 3
    div_num_c = target.shape[2] // div_c if process_extra_spectral_as_bgr else 1
    recon = np.empty((target.shape[0], target.shape[1], div_num_c * div_c), dtype=np.uint16)
    for div_idx_h in range(div_num_h):
        for div_idx_w in range(div_num_w):
            slice_end_h = (div_h * (div_idx_h + 1)) if div_idx_h != div_num_h - 1 else None
            slice_end_w = (div_w * (div_idx_w + 1)) if div_idx_w != div_num_w - 1 else None
            for div_idx_c in range(div_num_c):
                im_path = osp.join(out_dir, org_filename + f'_Div{div_idx_h}_{div_idx_w}_{div_idx_c}_output.png')
                recon[div_h * div_idx_h: slice_end_h,
                      div_w * div_idx_w: slice_end_w,
                      div_c * div_idx_c: div_c * (div_idx_c + 1)] = \
                    cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
                os.remove(im_path)
    if with_zeros:
        bin_mask = int('1' * in_bits + '0' * (16 - in_bits), 2)
    else:
        bin_mask = int('1' * (in_bits + 3) + '0' * (13 - in_bits), 2)

    recon = np.concatenate((recon, target[:, :, (div_num_c * div_c):] & bin_mask), 2)
    error = np.abs(recon.astype(np.int32) - target.astype(np.int32))
    np.savez_compressed(osp.join(out_dir, org_filename + '_error.npz'), a=error)
    if save_recon_tif:
        write_tiff_with_gdal(osp.join(out_dir, org_filename + '_recon.tif'), recon.transpose((2, 0, 1)))
        with zipfile.ZipFile(osp.join(out_dir, org_filename + '_recon.tif.zip'),
                             'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zip_f:
            zip_f.write(osp.join(out_dir, org_filename + '_recon.tif'), org_filename + '_recon.tif')
        os.remove(osp.join(out_dir, org_filename + '_recon.tif'))

    return target, recon


def cal_psnr_with_div(file_path, out_dir, in_bits, div_num_h, div_num_w, peak_value,
                      with_zeros, process_extra_spectral_as_bgr):
    target, recon = read_from_div(file_path, out_dir, in_bits, div_num_h, div_num_w,
                                  with_zeros, process_extra_spectral_as_bgr)
    bgr_psnr = 10 * np.log10(peak_value ** 2 / np.mean(
        (recon[:, :, :3].astype(np.float64) - target[:, :, :3]) ** 2))
    psnr = 10 * np.log10(peak_value ** 2 / np.mean((recon.astype(np.float64) - target) ** 2))
    print(f'{osp.split(file_path)[1]} in_bits: {in_bits} :\n'
          f'    BGR PSNR {bgr_psnr.item()}\n'
          f'    PSNR {psnr.item()}')

    return bgr_psnr.item(), psnr.item()


def test_ABCD(in_files, in_bits_range, peak_value, with_zeros, process_extra_spectral_as_bgr, model='edsr'):
    div_num_h, div_num_w = 8, 8
    div_dir = 'datasets/roots/RSMS/div_ABCD' # To be revised
    out_dir = f'result/save_{model}/{{in_bits}}-{16 if with_zeros else 13}'

    gen_bgr_div(in_files, div_dir, div_num_h, div_num_w, with_zeros, process_extra_spectral_as_bgr)

    bgr_psnr_dict = {}
    psnr_dict = {}
    if model == 'edsr':
        post_command = " --model save/edsr-abcd.pth"
    elif model == 'swin':
        post_command = " --model save/swin_abcd.pth --window 8"
    else: 
        raise NotImplemented(model)
    for in_bits in range(*in_bits_range):
        command = f"{sys.executable} test.py" \
                  f" --config configs/test_ABCD/abcd_test-16bits.yaml" \
                  f" --testset_root {div_dir}" \
                  f" --save_path {out_dir.format(in_bits=in_bits)}" \
                  f" --LBD {in_bits} --HBD {16 if with_zeros else 13} --gpu 0 --save 1"
        command += post_command
        subp = subprocess.run(command, shell=True, check=True, text=True)

        bgr_psnr_ls = []
        psnr_ls = []
        for file_path in in_files:
            bgr_psnr, psnr = cal_psnr_with_div(
                file_path, out_dir.format(in_bits=in_bits),
                in_bits, div_num_h, div_num_w, peak_value, with_zeros, process_extra_spectral_as_bgr)
            bgr_psnr_ls.append(bgr_psnr)
            psnr_ls.append(psnr)
        bgr_psnr_dict[in_bits] = bgr_psnr_ls
        psnr_dict[in_bits] = psnr_ls
    print('Done')

    return bgr_psnr_dict, psnr_dict


def test_bitmore(in_files, in_bits_range, peak_value, with_zeros, process_extra_spectral_as_bgr):
    div_num_h, div_num_w = 2, 2
    div_dir = 'data/Test/RSMS/div_bitmore' # To be revised
    out_dir = f'results/D16_quant_{{in_bits}}_{16 if with_zeros else 13}/RSMS/div_bitmore'

    gen_bgr_div(in_files, div_dir, div_num_h, div_num_w, with_zeros, process_extra_spectral_as_bgr)

    bgr_psnr_dict = {}
    psnr_dict = {}
    for in_bits in range(*in_bits_range):
        command = f"{sys.executable} test.py" \
                  f" --set_names {div_dir[len('data/Test/'):]}" \
                  f" --type_8_or_16 1 --quant {in_bits} --quant_end {16 if with_zeros else 13}" \
                  f" --dep 16 --save_result 1"
        subp = subprocess.run(command, shell=True, check=True, text=True)

        bgr_psnr_ls = []
        psnr_ls = []
        for file_path in in_files:
            bgr_psnr, psnr = cal_psnr_with_div(
                file_path, out_dir.format(in_bits=in_bits),
                in_bits, div_num_h, div_num_w, peak_value, with_zeros, process_extra_spectral_as_bgr)
            bgr_psnr_ls.append(bgr_psnr)
            psnr_ls.append(psnr)
        bgr_psnr_dict[in_bits] = bgr_psnr_ls
        psnr_dict[in_bits] = psnr_ls
    print('Done')

    return bgr_psnr_dict, psnr_dict


def test_ABCD_bitmore():
    peak_value = 10000
    process_extra_spectral_as_bgr = False

    if osp.exists('configs/test_ABCD'):
        in_bits_range = (4, 16)

        in_files = sorted(glob('datasets/roots/RSMS/**/*.tif', recursive=True)) # To be revised
        bgr_psnr_dict, psnr_dict = test_ABCD(in_files, in_bits_range, peak_value, True,
                                             process_extra_spectral_as_bgr, model='edsr')

        f = open('test_ABCD_edsr.csv', 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['Name', *(osp.split(_)[1] for _ in in_files)])
        for in_bits, value in psnr_dict.items():
            writer.writerow([f'ABCD {in_bits}({in_bits - 3}) -> 16', *value])
        for in_bits, value in bgr_psnr_dict.items():
            writer.writerow([f'ABCD {in_bits}({in_bits - 3}) -> 16 (BGR PSNR)', *value])
        f.close()

        bgr_psnr_dict, psnr_dict = test_ABCD(in_files, in_bits_range, peak_value, True,
                                             process_extra_spectral_as_bgr, model='swin')

        f = open('test_ABCD_swin.csv', 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['Name', *(osp.split(_)[1] for _ in in_files)])
        for in_bits, value in psnr_dict.items():
            writer.writerow([f'ABCD {in_bits}({in_bits - 3}) -> 16', *value])
        for in_bits, value in bgr_psnr_dict.items():
            writer.writerow([f'ABCD {in_bits}({in_bits - 3}) -> 16 (BGR PSNR)', *value])
        f.close()
    else:
        in_bits_range = (4, 16)

        in_files = sorted(glob('data/Test/RSMS/**/*.tif', recursive=True)) # To be revised
        bgr_psnr_dict, psnr_dict = test_bitmore(in_files, in_bits_range, peak_value, True,
                                                process_extra_spectral_as_bgr)

        f = open('test_bitmore.csv', 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['Name', *(osp.split(_)[1] for _ in in_files)])
        for in_bits, value in psnr_dict.items():
            writer.writerow([f'bitmore {in_bits}({in_bits - 3}) -> 16', *value])
        for in_bits, value in bgr_psnr_dict.items():
            writer.writerow([f'bitmore {in_bits}({in_bits - 3}) -> 16 (BGR PSNR)', *value])
        f.close()


def cal_error_dist():
    file_path = 'result/save/13-16/' \
                'TRIPLESAT_2_MS_L1_20191107021947_001FFCVI_002_0120200811001001_001.tif_error.npz'
    error = np.load(file_path)['a']
    h, w, c = error.shape
    count = np.sum(error == 0)
    print(f"Number of no error elements: {count}, {100 * count / (c * h * w)}%")
    count = np.sum(error == 1)
    print(f"Number of error-1 elements: {count}, {100 * count / (c * h * w)}%")
    for a in range(5):
        count = np.sum((error > 2 ** a) & (error <= 2 ** (a + 1)))
        print(f"Number of elements between {2 ** a} and {2 ** (a + 1)}: {count}, {100 * count / (c * h * w)}%")
    count = np.sum(error > 2 ** 5)
    print(f"Number of error>32 elements: {count}, {100 * count / (c * h * w)}%")


if __name__ == '__main__':
    # Place this script in the root directory of ABCD or bitmore
    test_ABCD_bitmore()
