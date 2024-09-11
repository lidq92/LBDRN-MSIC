import os
import csv
import subprocess
import numpy as np
from osgeo import gdal


gdal.UseExceptions()


def write_tiff_with_gdal(output_path: str, array: np.array):
    # The numpy array must be data with shape CHW
    gdal_type_mapping = { # Can add more if needed
        np.uint8: gdal.GDT_Byte,
        np.uint16: gdal.GDT_UInt16,
        np.float32: gdal.GDT_Float32,
        np.float64: gdal.GDT_Float64
    }
    gdal_dtype = gdal_type_mapping.get(array.dtype.type)
    if gdal_dtype is None: raise ValueError('Unsupported data type in this function!')
    driver = gdal.GetDriverByName('GTiff')
    # W, H, C
    dataset = driver.Create(output_path, array.shape[2], array.shape[1], array.shape[0], gdal_dtype)
    for i in range(array.shape[0]):
        band = dataset.GetRasterBand(i + 1)  
        band.WriteArray(array[i, :, :])
    dataset.FlushCache()
    dataset = None


def cleanup_temp_files(files):
    for file in files:
        if os.path.exists(file): os.remove(file)


def run_subprocess_commands(commands):
    for cmd in commands: subprocess.call(cmd, shell=True)


def encode(file_path, bin_path, method='JPEG2000star', K=1, q=None, d=None):
    if method in ['JPEG2000star', 'Baseline']: # lossless MSB + lossy LSB (q=2*K) or lossless MSB
        img = gdal.Open(file_path).ReadAsArray() # CHW
        MSB = img >> K
        MSB = MSB.astype(np.uint16) if MSB.max() > 255 else MSB.astype(np.uint8)
        write_tiff_with_gdal('JPEG2000_MSB.tif', MSB)

        run_subprocess_commands([
            f'gdal_translate -of JP2OpenJPEG -co QUALITY=100 -co REVERSIBLE=YES JPEG2000_MSB.tif JPEG2000_MSB.jp2',
        ])
        byte_to_write   = b''
        n_bytes_header = 6 if method == 'JPEG2000star' else 2
        byte_to_write += n_bytes_header.to_bytes(1, byteorder='big', signed=False)
        if method == 'JPEG2000star':
            MSB_bytes = os.path.getsize('JPEG2000_MSB.jp2')
            byte_to_write += MSB_bytes.to_bytes(4, byteorder='big', signed=False)
        byte_to_write += K.to_bytes(1, byteorder='big', signed=False)
        with open('JPEG2000star.header', 'wb') as fout: fout.write(byte_to_write)
        run_subprocess_commands([
            f'rm -f {bin_path}', 
            f'cat JPEG2000star.header >> {bin_path}',
            f'rm -f JPEG2000star.header',
            f'cat JPEG2000_MSB.jp2 >> {bin_path}',
            f'rm -f JPEG2000_MSB.jp2 JPEG2000_MSB.jp2.aux.xml JPEG2000_MSB.tif',
        ])
        if method == 'JPEG2000star':
            LSB = img - (MSB.astype(np.uint16) << K) #
            LSB = LSB.astype(np.uint16) if LSB.max() > 255 else LSB.astype(np.uint8)
            write_tiff_with_gdal('JPEG2000_LSB.tif', LSB)
            if q is None: q = 2 * K
            run_subprocess_commands([
                f'gdal_translate -of JP2OpenJPEG -co QUALITY={q} JPEG2000_LSB.tif JPEG2000_LSB.jp2',
                f'cat JPEG2000_LSB.jp2 >> {bin_path}',
                f'rm -f JPEG2000_LSB.jp2 JPEG2000_LSB.jp2.aux.xml JPEG2000_LSB.tif',
            ])
    if method == 'JPEG2000': # JP2OpenJPEG driver with gdal_translate
        # A whole image as a tile (by setting BLOCKXSIZE and BLOCKYSIZE) should prodcue better compression performance.
        # qs = [43.5, 33.5, 28, 22, 16, 11.5] # [100 / r for r in [2.3, 3, 3.6, 4.5, 6.2, 8.8]]
        qs = [43.5, 33.5, 28, 22, 16, 11.5, 10, 8, 6, 4, 2]
        if q is None: q = qs[K-1]
        run_subprocess_commands([
            f'gdal_translate -of JP2OpenJPEG -co QUALITY={q} {file_path} {bin_path}',
            f'rm -f {bin_path}.aux.xml',
        ])
    if method == 'JPEGXL': #libjxl
        # or https://gdal.org/en/latest/drivers/raster/jpegxl.html, 
        # https://gdal.org/en/latest/drivers/raster/gtiff.html#raster-gtiff
        ds = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24] 
        if d is None: d = ds[K-1]
        effort = 7 # 10 # 1 #
        dataset = gdal.Open(file_path)
        band_count = dataset.RasterCount
        for b in range(band_count):
            run_subprocess_commands([
                f'gdal_translate -b {b+1} {file_path} band{b+1}.png',
                f'cjxl band{b+1}.png {bin_path}{b+1} -e {effort} -d {d}',
                f'rm -f band{b+1}.png band{b+1}.png.aux.xml',
            ])
        byte_to_write   = b''
        n_bytes_header = 2 + 4 * (band_count - 1)
        byte_to_write += n_bytes_header.to_bytes(1, byteorder='big', signed=False)
        byte_to_write += band_count.to_bytes(1, byteorder='big', signed=False)
        for b in range(band_count-1):
            band_bytes = os.path.getsize(f'{bin_path}{b+1}')
            byte_to_write += band_bytes.to_bytes(4, byteorder='big', signed=False)
        with open(f'{method}.header', 'wb') as fout: fout.write(byte_to_write)
        run_subprocess_commands([
            f'rm -f {bin_path}', 
            f'cat {method}.header >> {bin_path}',
            f'rm -f {method}.header',
        ])
        for b in range(band_count):
            run_subprocess_commands([
                f'cat {bin_path}{b+1} >> {bin_path}',
                f'rm -f {bin_path}{b+1}',
            ])


def decode(bin_path, recon_path, method='JPEG2000star'):
    if method in ['JPEG2000star', 'Baseline']:
        with open(bin_path, 'rb') as fin: bitstream = fin.read()
        ptr = 0
        n_bytes_header = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
        ptr += 1
        MSB_bytes = None
        if method == 'JPEG2000star':
            MSB_bytes = int.from_bytes(bitstream[ptr: ptr + 4], byteorder='big', signed=False)
            ptr += 4
        K =  int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
        ptr += 1
        bitstream = bitstream[ptr:]
        if method == 'JPEG2000star':
            with open('JPEG2000_MSB.jp2', 'wb') as f_out: f_out.write(bitstream[:MSB_bytes])
            with open('JPEG2000_LSB.jp2', 'wb') as f_out: f_out.write(bitstream[MSB_bytes:])
            run_subprocess_commands([
                f'gdal_translate -of GTiff JPEG2000_MSB.jp2 JPEG2000_MSB.tif',
                f'gdal_translate -of GTiff JPEG2000_LSB.jp2 JPEG2000_LSB.tif',
            ])
        else:
            with open('JPEG2000_MSB.jp2', 'wb') as f_out: f_out.write(bitstream)
            run_subprocess_commands([
                f'gdal_translate -of GTiff JPEG2000_MSB.jp2 JPEG2000_MSB.tif',
            ])
            
        MSB = gdal.Open('JPEG2000_MSB.tif').ReadAsArray() # CHW
        LSB = gdal.Open('JPEG2000_LSB.tif').ReadAsArray() if method == 'JPEG2000star' else np.zeros_like(MSB)
        img = (MSB.astype(np.uint16) << K) + LSB.astype(np.uint16)
        write_tiff_with_gdal(recon_path, img)
        cleanup_temp_files(['JPEG2000_MSB.jp2', 'JPEG2000_MSB.tif', 'JPEG2000_LSB.jp2', 'JPEG2000_LSB.tif'])
    if method == 'JPEG2000':
        subprocess.call(f"gdal_translate -of GTiff {bin_path} {recon_path}", shell=True)
    if method == 'JPEGXL':
        with open(bin_path, 'rb') as fin: bitstream = fin.read()
        ptr = 0
        n_bytes_header = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
        ptr += 1
        band_count =  int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
        ptr += 1
        band_bytes = [0 for b in range(band_count-1)]
        for b in range(band_count-1):
            band_bytes[b] = int.from_bytes(bitstream[ptr: ptr + 4], byteorder='big', signed=False)
            ptr += 4
        bitstream = bitstream[ptr:]
        for b in range(band_count):
            with open(f'{bin_path}{b+1}', 'wb') as f_out: 
                if b < band_count - 1:
                    f_out.write(bitstream[:band_bytes[b]])
                    bitstream = bitstream[band_bytes[b]:]
                else:
                    f_out.write(bitstream)
            subprocess.call(f"djxl {bin_path}{b+1} band{b+1}.png", shell=True)
        cmd = f'gdalbuildvrt -separate {recon_path}.vrt'
        for b in range(band_count):
            cmd += f' band{b+1}.png'
        run_subprocess_commands([
            cmd,
            f'gdal_translate {recon_path}.vrt {recon_path}',
            f'rm -f {recon_path}.vrt',
        ])
        for b in range(band_count):
            subprocess.call(f'rm -f {bin_path}{b+1} band{b+1}.png', shell=True)


def eval_RD(bin_path, recon_path, file_path):
    org_img = gdal.Open(file_path).ReadAsArray() # CHW
    recon_img = gdal.Open(recon_path).ReadAsArray() # CHW
    mse_value = np.mean((org_img.astype(np.float32) - recon_img.astype(np.float32)) ** 2) #
    peak = 10000 # np.max(org_img) # 65535 #
    psnr = 10 * np.log10(peak ** 2 / mse_value)
    bits = 8 * os.path.getsize(bin_path)
    C, H, W = org_img.shape
    bpsp = bits / (C * H * W)
    print(f"MSE: {mse_value}, PSNR: {psnr}, bits: {bits}, bpsp: {bpsp}")

    return mse_value, psnr, bits, bpsp


def main():
    file_paths = [
        "data/GF-dataset/GF-2/TRIPLESAT_2_MS_L1_20191107021947_001FFCVI_002_0120200811001001_001.tif",
        "data/GF-dataset/GF-2/TRIPLESAT_2_MS_L1_20191107021950_001FFCVI_003_0120200811001001_002.tif",
        "data/GF-dataset/GF-2/TRIPLESAT_2_MS_L1_20191107021954_001FFCVI_004_0120200811001001_001.tif",
        "data/GF-dataset/GF-2/TRIPLESAT_2_MS_L1_20200109023258_002115VI_003_0120200811001001_001.tif",
        "data/GF-dataset/GF-2/TRIPLESAT_2_MS_L1_20200109023301_002115VI_004_0120200811001001_001.tif",
        "data/GF-dataset/GF-6/GF6-WFI/GF6_WFI_Sample_A.tif",
        "data/GF-dataset/GF-6/GF6-WFI/GF6_WFI_Sample_B.tif",
        "data/GF-dataset/GF-6/GF6-WFI/GF6_WFI_Sample_C.tif",
        "data/GF-dataset/GF-6/GF6-WFI/GF6_WFI_Sample_D.tif",
        "data/GF-dataset/GF-6/GF6-PMS/GF6_PMS_Sample_A.tif",
        "data/GF-dataset/GF-6/GF6-PMS/GF6_PMS_Sample_B.tif",
        "data/GF-dataset/GF-6/GF6-PMS/GF6_PMS_Sample_C.tif",
        "data/GF-dataset/GF-6/GF6-PMS/GF6_PMS_Sample_D.tif",
    ]

    methods = [
        'JPEGXL', 
        'JPEG2000', 
        'JPEG2000star',
        'Baseline',
        ]
    os.makedirs('SOTA_results', exist_ok=True)
    for method in methods:
        csv_file = f'SOTA_results/{method}_11rps.csv'
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            metrics = ['MSE', 'PSNR', 'bpsp', 'bits']
            csv_headers = ['K'] + [f"{path}_{metric}" for path in file_paths for metric in metrics]
            writer.writerow(csv_headers)
            # for K in range(1, 7):
            for K in range(1, 12):
                row = [f"K{K}"] + [None] * (len(file_paths) * 4)
                for i, file_path in enumerate(file_paths):
                    bin_path = f'{method}_{i}_{K}.bin'
                    encode(file_path, bin_path, method, K)
                    recon_path = f'{method}_{i}_{K}.tif'
                    decode(bin_path, recon_path, method)
                    mse_value, psnr, bits, bpsp = eval_RD(bin_path, recon_path, file_path)
                    row[4*i+1] = mse_value
                    row[4*i+2] = psnr
                    row[4*i+3] = bpsp
                    row[4*i+4] = bits
                    if True: subprocess.call(f'rm -f {bin_path} {recon_path}', shell=True)   
                writer.writerow(row)


def error_reconstruction():
    file_path = "data/GF-dataset/GF-2/TRIPLESAT_2_MS_L1_20191107021947_001FFCVI_002_0120200811001001_001.tif"

    method = 'Baseline'
    bin_path = f'{method}.bin'
    encode(file_path, bin_path, method, K=4)
    recon_path = f'{method}.tif'
    decode(bin_path, recon_path, method)
    _, psnr, _, bpsp = eval_RD(bin_path, recon_path, file_path)
    subprocess.call(f'rm -f {bin_path}', shell=True)     
    msg = error_stats(file_path, recon_path)     
    msg = f'Baseline & ${round(bpsp,3):.3f}$ & ${round(psnr,3):.3f}$ & {msg}'
    print(msg)
    # subprocess.call(f'rm -f {recon_path}', shell=True)  

    method = 'JPEG2000star'
    bin_path = f'{method}.bin'
    encode(file_path, bin_path, method, K=5, q=8.84)
    recon_path = f'{method}.tif'
    decode(bin_path, recon_path, method)
    _, psnr, _, bpsp = eval_RD(bin_path, recon_path, file_path)
    subprocess.call(f'rm -f {bin_path}', shell=True)     
    msg = error_stats(file_path, recon_path)     
    msg = f'JPEG 2000$^*$ & ${round(bpsp,3):.3f}$ & ${round(psnr,3):.3f}$ & {msg}'
    print(msg)
    # subprocess.call(f'rm -f {recon_path}', shell=True)   

    method = 'JPEG2000'
    bin_path = f'{method}.bin'
    encode(file_path, bin_path, method, q=18.66)
    recon_path = f'{method}.tif'
    decode(bin_path, recon_path, method)
    _, psnr, _, bpsp = eval_RD(bin_path, recon_path, file_path)
    subprocess.call(f'rm -f {bin_path}', shell=True)     
    msg = error_stats(file_path, recon_path)     
    msg = f'JPEG 2000 & ${round(bpsp,3):.3f}$ & ${round(psnr,3):.3f}$ & {msg}'
    print(msg)
    # subprocess.call(f'rm -f {recon_path}', shell=True)    
      
    method = 'JPEGXL'
    bin_path = f'{method}.bin'
    encode(file_path, bin_path, method, d=0.0133)
    recon_path = f'{method}.tif'
    decode(bin_path, recon_path, method)
    _, psnr, _, bpsp = eval_RD(bin_path, recon_path, file_path)
    subprocess.call(f'rm -f {bin_path}', shell=True)     
    msg = error_stats(file_path, recon_path)     
    msg = f'JPEG XL & ${round(bpsp,3):.3f}$ & ${round(psnr,3):.3f}$ & {msg}'
    print(msg)
    # subprocess.call(f'rm -f {recon_path}', shell=True)              
    

def error_stats(file_path, recon_path):
    ori_img = gdal.Open(file_path).ReadAsArray() # CHW
    recon_img = gdal.Open(recon_path).ReadAsArray() # CHW
    error = np.abs(ori_img.astype(np.float32) - recon_img.astype(np.float32))
    C, H, W = error.shape
    msg = f''
    count = np.sum(error == 0)
    # print(f"Number of no error elements: {count}, {round(100 * count / (C * H * W),3):.3f}%")
    msg += f"${round(100 * count / (C * H * W),3):.3f}\\%$ & "
    count = np.sum(error == 1)
    # print(f"Number of error-1 elements: {count}, {round(100 * count / (C * H * W),3):.3f}%")
    msg += f"${round(100 * count / (C * H * W),3):.3f}\\%$ & "
    for a in range(4):
        count = np.sum((error > 2 ** a) & (error <= 2 ** (a+1)))
        # print(f"Number of elements between {2 ** a} and {2 ** (a+1)}: {count}, {round(100 * count / (C * H * W),3):.3f}%")
        msg += f"${round(100 * count / (C * H * W),3):.3f}\\%$ & "
    count = np.sum(error > 2 ** 4)
    # print(f"Number of error > 16 elements: {count}, {round(100 * count / (C * H * W),3):.3f}%")
    msg += f"${round(100 * count / (C * H * W),3):.3f}\\%$ \\\\"
    
    return msg


if __name__ == "__main__":
    main()
    error_reconstruction()