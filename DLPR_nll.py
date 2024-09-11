# Compression of GF MSI using DLPR + jxl
import os
import sys
import time
import struct
import shutil
import tempfile
import subprocess

import torch
import compressai
import numpy as np
import torch.nn.functional as F
from osgeo import gdal
from torchvision import transforms
from PIL import Image, PngImagePlugin
from compressai.zoo import image_models as pretrained_models

import logger

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ln -s abs_DLPR_PATH Deep-Lossy-Plus-Residual-Coding/DLPR_nll/
# ln -s abs_data_path xxx_as_below
DLPR_PATH = 'Deep-Lossy-Plus-Residual-Coding/DLPR_nll/'
WORKLIST = [    
        ['data/GF-dataset/GF-2/TRIPLESAT_2_MS_L1_20191107021947_001FFCVI_002_0120200811001001_001.tif', 4, 'tmpA.dir'],
        ['data/GF-dataset/GF-2/TRIPLESAT_2_MS_L1_20191107021950_001FFCVI_003_0120200811001001_002.tif', 4, 'tmpB.dir'], 
        ['data/GF-dataset/GF-2/TRIPLESAT_2_MS_L1_20191107021954_001FFCVI_004_0120200811001001_001.tif', 4, 'tmpC.dir'], 
        ['data/GF-dataset/GF-2/TRIPLESAT_2_MS_L1_20200109023258_002115VI_003_0120200811001001_001.tif', 4, 'tmpD.dir'], 
        ['data/GF-dataset/GF-2/TRIPLESAT_2_MS_L1_20200109023301_002115VI_004_0120200811001001_001.tif', 4, 'tmpE.dir'], 
        ['data/GF-dataset/GF-6/GF6-WFI/GF6_WFI_Sample_A.tif', 8, 'tmpDWA.dir'], 
        ['data/GF-dataset/GF-6/GF6-WFI/GF6_WFI_Sample_B.tif', 8, 'tmpDWB.dir'], 
        ['data/GF-dataset/GF-6/GF6-WFI/GF6_WFI_Sample_C.tif', 8, 'tmpDWC.dir'], 
        ['data/GF-dataset/GF-6/GF6-WFI/GF6_WFI_Sample_D.tif', 8, 'tmpDWD.dir'],         
        ['data/GF-dataset/GF-6/GF6-PMS/GF6_PMS_Sample_A.tif', 4, 'tmpDPA.dir'], 
        ['data/GF-dataset/GF-6/GF6-PMS/GF6_PMS_Sample_B.tif', 4, 'tmpDPB.dir'], 
        ['data/GF-dataset/GF-6/GF6-PMS/GF6_PMS_Sample_C.tif', 4, 'tmpDPC.dir'], 
        ['data/GF-dataset/GF-6/GF6-PMS/GF6_PMS_Sample_D.tif', 4, 'tmpDPD.dir'], 
    ]


def calculate_psnr(matrix_org, matrix_recon):
    mse_value = np.mean((matrix_org.astype(np.float32) - matrix_recon.astype(np.float32)) ** 2)
    peak = 10000
    psnr = 10*np.log10(peak**2/mse_value)

    return psnr


def write_floats(fd, values, fmt=">{:d}f"):
    fd.write(struct.pack(fmt.format(len(values)), *values))

    return len(values) * 4  # Assuming float is 4 bytes


def read_floats(fd, n, fmt=">{:d}f"):
    sz = struct.calcsize("f")

    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_ints(fd, values, fmt=">{:d}i"):
    fd.write(struct.pack(fmt.format(len(values)), *values))

    return len(values) * 4


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))

    return len(values) * 1


def read_ints(fd, n, fmt=">{:d}i"):
    sz = struct.calcsize("i")

    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")

    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0: return
    fd.write(struct.pack(fmt.format(len(values)), values))

    return len(values) * 1


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")

    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]

def read_body(fd):
    return read_bytes(fd, read_ints(fd, 1)[0])


def write_body(fd, out_strings):
    bytes_cnt = 0
    bytes_cnt += write_ints(fd, (len(out_strings),))
    bytes_cnt += write_bytes(fd, out_strings)

    return bytes_cnt


def readTif(input_file, num_channel=0):
    try:
        in_file = gdal.Open(input_file)
        if in_file == None:
            print(input_file + " can not be opened")
            return
    except:
        print("Open Tiff file error")
        return

    width = in_file.RasterXSize
    height = in_file.RasterYSize
    in_info = gdal.Info(in_file, format='json')

    bands = in_file.RasterCount                            

    if num_channel == 0:    # default 4 channel
        bandData_0 = in_file.GetRasterBand(0+1).ReadAsArray()
        bandData_1 = in_file.GetRasterBand(1+1).ReadAsArray()
        bandData_2 = in_file.GetRasterBand(2+1).ReadAsArray()
        bandData_3 = in_file.GetRasterBand(3+1).ReadAsArray()

        return bandData_0, bandData_1, bandData_2, bandData_3, width, height
    else:
        bandsData = []
        if num_channel <= bands:
            bandData_list = [in_file.GetRasterBand(i+1).ReadAsArray() for i in range(num_channel)]
            bandsData = np.stack(bandData_list, axis=-1)
        else:
            print("Open Tiff erro, num channel > bands")

        return bandsData, width, height


def read_tiff_with_gdal(input_file):
    try:
        in_file = gdal.Open(input_file)
        if in_file == None:
            print(input_file + " can not be opened")
            return None
    except:
        print("Open Tiff file error")
        return None

    bands = in_file.RasterCount
    bandData = None
    if bands > 1 :
        image_data = []
        for i in range(1, bands + 1):
            band_data = in_file.GetRasterBand(i).ReadAsArray()
            image_data.append(band_data)

        bandData = np.stack(image_data, axis=-1)
    else:
        bandData = in_file.GetRasterBand(0+1).ReadAsArray()

    return bandData


def split_uint16_to_uint8(array):
    max_val = np.max(array)
    min_val = np.min(array)
    max_bit = int(np.ceil(np.log2(max_val-min_val + 1)))
    shift_amount =16 - max_bit
    bias_array = (array - min_val) << shift_amount
    msb_array = (bias_array >> 8).astype(np.uint8)
    lsb_array = (bias_array & 0xFF).astype(np.uint8)
    
    return msb_array, lsb_array, shift_amount, min_val


def merge_uint8_to_uint16(msb_array, lsb_array, shift_amount, min_val):
    merged_array = (msb_array.astype(np.uint16) << 8) | lsb_array
    original_array = (merged_array >> shift_amount) + min_val
    
    return original_array.astype(np.uint16)


def uint16_to_uint8(array):
    min_val = array.min()
    max_val = array.max()
    scaled_array = ((array - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    return scaled_array, max_val, min_val


def uint8_to_uint16(scaled_array, max_val, min_val):
    original_array = (scaled_array.astype(np.float32) / 255 * (max_val - min_val) + min_val).astype(np.uint16)

    return original_array


# array (h, w, 3) or array (h, w, 1)
def write_tiff_with_gdal(output_path: str, array: np.array):
    # The numpy array must be data with shape CHW
    gdal.UseExceptions()
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
    dataset = driver.Create(output_path, array.shape[1], array.shape[0], array.shape[2], gdal_dtype)
    for i in range(array.shape[2]):
        band = dataset.GetRasterBand(i + 1)  
        band.WriteArray(array[:, :, i])
    dataset.FlushCache()
    dataset = None


def compress_jxl_lossy(image_data, distance):
    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp_image_file:
        temp_image_path = temp_image_file.name
        # write_tiff_with_gdal(temp_image_path, image_data)
        image = Image.fromarray(image_data)
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text("bitdepth", str(image_data.dtype.itemsize * 8))  
        image.save(temp_image_path, format='PNG', pnginfo=png_info)
    
    # Define temporary file for compressed output
    with tempfile.NamedTemporaryFile(suffix='.jxl', delete=False) as temp_compressed_file:
        temp_compressed_path = temp_compressed_file.name
    
    command = [
        'cjxl',
        temp_image_path,    # Use temporary file as input
        temp_compressed_path,  # Use temporary file as output
        '-d', str(distance),
        '-e', '9'
    ]
    
    try:
        print(f'command {command}')
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("Jxl lossy compression completed.")
            with open(temp_compressed_path, 'rb') as compressed_file:
                compressed_data = compressed_file.read()
                compressed_size = len(compressed_data)
            return temp_compressed_path, compressed_size
        else:
            print(f"Jxl lossy compression failed. Error: {result.stderr.decode('utf-8')}")
            return None, None
    finally:
        # Clean up: remove temporary files
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)      
        # print()   


def decompress_jxl(compressed_file_path):
    # Define temporary file for decompressed output
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_decompressed_file:
        temp_decompressed_path = temp_decompressed_file.name
    
    command = [
        'djxl',
        compressed_file_path,    # Input compressed file
        temp_decompressed_path   # Output decompressed file
    ]
    
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("Jxl decompression completed.")
            # Read the decompressed image data
            # Adjust this part based on how your decompressed output is handled
            decompressed_image_data = read_tiff_with_gdal(temp_decompressed_path)  # Replace with your actual reading function
            # Ensure the shape matches the original
            return decompressed_image_data
        
        else:
            print(f"Jxl decompression failed. Error: {result.stderr.decode('utf-8')}")
            return None
    finally:
        # Clean up: remove temporary files
        if os.path.exists(temp_decompressed_path):
            os.remove(temp_decompressed_path)  
        if os.path.exists(compressed_file_path):
            os.remove(compressed_file_path)      


# image_data (h, w, 3)
def compress_dlpr_lossy(nll_model, image_data, COT, tau):
    sys.path.insert(0, DLPR_PATH) #
    from nll_model_eval import NearLosslessCompressor
    from nll_test import coding_order_table7x7, compress, decompress

    I = np.array(image_data).astype(np.float32)
    code_lossy, code_res, img_shape, res_range = compress(nll_model, I, COT, tau)    

    # Define temporary file for compressed output
    with tempfile.NamedTemporaryFile(suffix='.dlpr', delete=False) as temp_compressed_file:
        temp_compressed_path = temp_compressed_file.name

    compressed_size = 0
    with open(temp_compressed_path, 'wb') as fout:
        code_lossy_sz = len(code_lossy['img_strings'][0])
        write_ints(fout, (code_lossy_sz,))
        for i in range(code_lossy_sz):
            write_body(fout, code_lossy['img_strings'][0][i])
            write_body(fout, code_lossy['img_strings'][1][i])
        write_ints(fout, code_lossy['shape'])
        code_res_sz = len(code_res)
        write_ints(fout, (code_res_sz,))
        for i in range(code_res_sz):
            write_body(fout, code_res[i])
        write_ints(fout, res_range)
        
    with open(temp_compressed_path, 'rb') as compressed_file:
        compressed_data = compressed_file.read()
        compressed_size = len(compressed_data)

    return temp_compressed_path, compressed_size


def decompress_dlpr_lossy(nll_model, compressed_file_path, img_shape, COT, tau):
    sys.path.insert(0, DLPR_PATH) #
    from nll_model_eval import NearLosslessCompressor
    from nll_test import coding_order_table7x7, compress, decompress

    code_lossy = {
    'img_strings': [[], []],  
    'shape': [],  
    }
    code_res = []  
    res_range = []

    try:
        with open(compressed_file_path, 'rb') as fin:  
            code_lossy_sz = read_ints(fin, 1)[0]
            for i in range(code_lossy_sz):
                code_lossy['img_strings'][0].append(read_body(fin))
                code_lossy['img_strings'][1].append(read_body(fin))
            code_lossy['shape'] = read_ints(fin, 4)
            
            code_res_sz = read_ints(fin, 1)[0]
            for i in range(code_res_sz):
                code_res.append(read_body(fin))
            
            res_range = read_ints(fin, 2)
    finally:
        # Clean up: remove temporary files
        if os.path.exists(compressed_file_path):
            os.remove(compressed_file_path)    

    return decompress(nll_model, code_lossy, code_res, img_shape, res_range, COT, tau).cpu().numpy().astype(np.uint8)


global_nll_model = None
def get_nll_model():
    sys.path.insert(0, DLPR_PATH)
    from nll_model_eval import NearLosslessCompressor
    from nll_test import coding_order_table7x7, compress, decompress    

    global global_nll_model
    COT = coding_order_table7x7()

    if global_nll_model is None:
        ckp_dir = f"{DLPR_PATH}/ckp_nll"

        device = torch.device('cuda')
        nll_model = NearLosslessCompressor(192, 5).eval().to(device)

        ckp = torch.load(os.path.join(ckp_dir, "ckp.tar"), map_location=device)
        nll_model.load_state_dict(ckp['model_state_dict'])
        nll_model.lossy_compressor.update(force=True)
        
        global_nll_model = nll_model
    else:
        nll_model = global_nll_model
        

    return nll_model, COT


def split_large_matrix(matrix, height, width, block_size):
    max_block_height, max_block_width = block_size
    if matrix is not None:
        matrix = matrix.reshape(height, width)

    blocks = []
    block_sizes = []  
    
    for y in range(0, height, max_block_height):
        for x in range(0, width, max_block_width):
            block_end_y = min(y + max_block_height, height)
            block_end_x = min(x + max_block_width, width)
            
            if matrix is not None:
                block = matrix[y:block_end_y, x:block_end_x]
                blocks.append(block)
            
            block_height = block_end_y - y
            block_width = block_end_x - x
            block_sizes.append((block_height, block_width, y, x))
    
    return blocks, block_sizes


def save_blocks_one_file(output_filename, compress_method, compressed_blocks_info, 
                         height, width, compress_level, num_channel):
    num_write_bytes = 0
    num_split = int(len(compressed_blocks_info) / (num_channel-2))

    #save header
    with open(output_filename, 'wb') as fout:
        num_write_bytes += write_ints(fout, (height, width, num_channel, num_split))
        num_write_bytes += write_floats(fout, (compress_level,))
        write_ints_lst = []
        for i in range(num_split):
            nIdx = i * (num_channel-2)
            msb_compressed_length = compressed_blocks_info[nIdx][1]
            band0_block_max_shift = compressed_blocks_info[nIdx][2]
            band0_block_min = compressed_blocks_info[nIdx][3]
            band1_block_max_shift = compressed_blocks_info[nIdx][4]
            band1_block_min = compressed_blocks_info[nIdx][5]
            band2_block_max_shift = compressed_blocks_info[nIdx][6]
            band2_block_min = compressed_blocks_info[nIdx][7]

            msb_ints_lst = [msb_compressed_length, band0_block_max_shift, band0_block_min, band1_block_max_shift, band1_block_min, band2_block_max_shift, band2_block_min]
            write_ints_lst.extend(msb_ints_lst)
            for j in range(1, num_channel-2):
                compressed_length = compressed_blocks_info[nIdx+j][1]
                write_ints_lst.append(compressed_length)
            
        num_write_bytes += write_ints(fout, write_ints_lst)

    # save file content 
    for i in range(num_split):
        nIdx = i * (num_channel-2)
        msb_compressed_file = compressed_blocks_info[nIdx][0]
        subprocess.call(f'cat {msb_compressed_file} >> {output_filename}', shell=True)
        subprocess.call(f'rm -f {msb_compressed_file}', shell=True)
        num_write_bytes += compressed_blocks_info[nIdx][1]

        for j in range(1, num_channel-2):
            compressed_file = compressed_blocks_info[nIdx+j][0]
            subprocess.call(f'cat {compressed_file} >> {output_filename}', shell=True)
            subprocess.call(f'rm -f {compressed_file}', shell=True)
            num_write_bytes += compressed_blocks_info[nIdx+j][1]

    return num_write_bytes


def encode_big_file(input_filename, output_filename, compress_method, compress_level, block_size, num_channel):
    '''
    compress_method : jxl-dlpr-linear, jxl-dlpr-msb
    compress_level : tau at jxl-dlpr-linear, jxl-dlpr-msb
    '''

    org_img, width, height = readTif(input_filename, num_channel)

    if compress_method == "jxl-dlpr-linear" or compress_method == "jxl-dlpr-msb":
        nll_model, COT = get_nll_model()
        band_block_info = []
        for i in range(num_channel):
            band_blocks, band_block_sizes = split_large_matrix(org_img[:,:,i], height, width, block_size)
            band_block_info.append((band_blocks, band_block_sizes))
        
        compressed_blocks_info = []
        num_split = len(band_block_info[0][0])   # split nums / len band_balcks
        for i in range(num_split): 
            if compress_method == "jxl-dlpr-linear" :
                band0_block_msb, band0_block_max, band0_block_min = uint16_to_uint8(band_block_info[0][0][i])
                band1_block_msb, band1_block_max, band1_block_min = uint16_to_uint8(band_block_info[1][0][i])
                band2_block_msb, band2_block_max, band2_block_min = uint16_to_uint8(band_block_info[2][0][i])
                msb_block_img = np.stack((band2_block_msb, band1_block_msb, band0_block_msb), axis=-1)
                
                msb_compressed_file, msb_compressed_length = compress_dlpr_lossy(nll_model, msb_block_img.reshape((band_block_info[0][1][i][0], band_block_info[0][1][i][1], 3)), COT, compress_level)
                compressed_blocks_info.append((msb_compressed_file, msb_compressed_length, band0_block_max, band0_block_min, band1_block_max, band1_block_min, band2_block_max, band2_block_min))
            
            elif compress_method == "jxl-dlpr-msb" :
                band0_block_msb, band0_block_lsb, band0_block_shift, band0_block_min = split_uint16_to_uint8(band_block_info[0][0][i])
                band1_block_msb, band1_block_lsb, band1_block_shift, band1_block_min = split_uint16_to_uint8(band_block_info[1][0][i])
                band2_block_msb, band2_block_lsb, band2_block_shift, band2_block_min = split_uint16_to_uint8(band_block_info[2][0][i])
                msb_block_img = np.stack((band2_block_msb, band1_block_msb, band0_block_msb), axis=-1)

                msb_compressed_file, msb_compressed_length = compress_dlpr_lossy(nll_model, msb_block_img.reshape((band_block_info[0][1][i][0], band_block_info[0][1][i][1], 3)), COT, compress_level)
                compressed_blocks_info.append((msb_compressed_file, msb_compressed_length, band0_block_shift, band0_block_min, band1_block_shift, band1_block_min, band2_block_shift, band2_block_min))
            
            else:
                print('compress method error at 1 !')            
            
            for j in range(3, num_channel):
                distance = compress_level * 0.015 + 0.045 
                band_compressed_file, band_compressed_length = compress_jxl_lossy(band_block_info[j][0][i].reshape((band_block_info[j][1][i][0], band_block_info[j][1][i][1])), distance)
                compressed_blocks_info.append((band_compressed_file, band_compressed_length))
        
        compressed_file_size = save_blocks_one_file(output_filename, compress_method, compressed_blocks_info, height, width, compress_level, num_channel)    
    
    return compressed_file_size, org_img, height, width


def gen_tmpfile_name():
    tmpfile_name = ''
    with tempfile.NamedTemporaryFile(suffix='.tmp', delete=False) as temp_file:
        tmpfile_name = temp_file.name

    return tmpfile_name


def create_file_from_stream(fin, filename, length):
    with open(filename, 'wb') as fout:
        fout.write(fin.read(length))


def parser_blocks_from_file(compressed_filename, compress_method):
    compressed_blocks_info = []
    height, width, num_channel, num_split = 0, 0, 0, 0
    compress_level = 0.0
    with open(compressed_filename, 'rb') as fin:
        height, width, num_channel, num_split = read_ints(fin, 4)
        compress_level = read_floats(fin, 1)[0]
        for i in range(num_split):
            nIdx = i * (num_channel-2)
            msb_compressed_file  = gen_tmpfile_name()
            msb_compressed_length, band0_block_max_shift, band0_block_min, band1_block_max_shift, band1_block_min, band2_block_max_shift, band2_block_min = read_ints(fin, 7)
            compressed_blocks_info.append((msb_compressed_file, msb_compressed_length, band0_block_max_shift, band0_block_min, band1_block_max_shift, band1_block_min, band2_block_max_shift, band2_block_min))
            for j in range(1, num_channel-2):
                compressed_file = gen_tmpfile_name()
                compressed_length = read_ints(fin, 1)[0]
                compressed_blocks_info.append((compressed_file, compressed_length))

        for i in range(num_split):
            nIdx = i * (num_channel-2)
            msb_compressed_file = compressed_blocks_info[nIdx][0]
            msb_compressed_length = compressed_blocks_info[nIdx][1]
            create_file_from_stream(fin, msb_compressed_file, msb_compressed_length)
            for j in range(1, num_channel-2):
                compressed_file = compressed_blocks_info[nIdx+j][0]
                compressed_length = compressed_blocks_info[nIdx+j][1]
                create_file_from_stream(fin, compressed_file, compressed_length)

    return compressed_blocks_info, height, width, num_channel, compress_level


def decode_big_file(compressed_filename, compress_method, block_size):
    img_recon = None

    if compress_method == "jxl-dlpr-linear" or compress_method == "jxl-dlpr-msb":
        nll_model, COT = get_nll_model()

        compressed_blocks_info, height, width, num_channel, compress_level = parser_blocks_from_file(compressed_filename, compress_method)
        blocks, block_sizes = split_large_matrix(None, height, width, block_size)
        img_recon = np.zeros((height, width, num_channel), dtype=np.uint16)
        if(len(block_sizes) != int(len(compressed_blocks_info)/(num_channel-2))):
            print('decode error!!!!')
            return None
        
        for i in range(len(block_sizes)):
            band_block_recon_lst = []
            block_height, block_width, y, x = block_sizes[i]
            nIdx = i * (num_channel-2)

            msb_compressed_file, msb_compressed_length, band0_block_max_shift, band0_block_min, band1_block_max_shift, band1_block_min, band2_block_max_shift, band2_block_min  = compressed_blocks_info[nIdx]
            msb_block_img_recon = decompress_dlpr_lossy(nll_model, msb_compressed_file, (block_height, block_width), COT, int(compress_level))
            band0_block_msb_recon = msb_block_img_recon[:, :, 2]
            band1_block_msb_recon = msb_block_img_recon[:, :, 1]
            band2_block_msb_recon = msb_block_img_recon[:, :, 0]
            if compress_method == "jxl-dlpr-linear":
                band0_block_recon = uint8_to_uint16(band0_block_msb_recon, band0_block_max_shift, band0_block_min)
                band1_block_recon = uint8_to_uint16(band1_block_msb_recon, band1_block_max_shift, band1_block_min)
                band2_block_recon = uint8_to_uint16(band2_block_msb_recon, band2_block_max_shift, band2_block_min)
            elif compress_method == "jxl-dlpr-msb":
                band0_block_lsb_recon = np.zeros_like(band0_block_msb_recon)
                band1_block_lsb_recon = np.zeros_like(band1_block_msb_recon)
                band2_block_lsb_recon = np.zeros_like(band2_block_msb_recon)
                band0_block_recon = merge_uint8_to_uint16(band0_block_msb_recon, band0_block_lsb_recon, band0_block_max_shift, band0_block_min)
                band1_block_recon = merge_uint8_to_uint16(band1_block_msb_recon, band1_block_lsb_recon, band1_block_max_shift, band1_block_min)
                band2_block_recon = merge_uint8_to_uint16(band2_block_msb_recon, band2_block_lsb_recon, band2_block_max_shift, band2_block_min)  
            else:
                print('compress method error at 3 !')    

            band_block_recon_lst.append(band0_block_recon)
            band_block_recon_lst.append(band1_block_recon)
            band_block_recon_lst.append(band2_block_recon)
            for j in range(1, num_channel-2):
                compressed_file, compressed_length = compressed_blocks_info[nIdx+j]
                band_block_recon = decompress_jxl(compressed_file)   
                band_block_recon_lst.append(band_block_recon)   

            block_img_recon = np.stack(band_block_recon_lst, axis=-1)
            img_recon[y:y+block_height, x:x+block_width] = block_img_recon[:block_height, :block_width]
    
        else:
            print('compress method error at 4 !')    

    return img_recon


def experiment_DL_dlpr_encode_big_file(item):
    # torch.cuda.set_device((4+item)%8)

    worklist = WORKLIST
    input_filename = worklist[item][0]
    num_channel = worklist[item][1]
    logger.create_logger(worklist[item][2])
    output_dir = worklist[item][2]
    os.makedirs(output_dir, exist_ok=True)
    block_size = (3000, 3000)
    logger.log.info(f"file name : {input_filename}")

    ##################################  dlpr  ############################################ 
    compress_method = 'jxl-dlpr-linear'         # jxl-dlpr-linear, jxl-dlpr-msb
    compress_levels = [1, 2, 3, 4, 5, 6, 8, 16, 32, 64, 128]        # mean tau (1~6) at jxl-dlpr-linear, jxl-dlpr-msb
    jxl_dlpr_linear = {"psnr": [], "bpsp": []}
    for compress_level in compress_levels:
        output_filename = os.path.join(output_dir, f"{compress_method}_{compress_level}.z")
        start_time = time.time()
        compressed_file_size, org_img, height, width = encode_big_file(input_filename, output_filename, compress_method, compress_level, block_size, num_channel)
        encode_time = time.time() - start_time
        start_time = time.time()
        recon_img = decode_big_file(output_filename, compress_method, block_size)
        decode_time = time.time() - start_time
        recon_filename = os.path.join(output_dir, f"{compress_method}_{compress_level}_rec.tif")
        write_tiff_with_gdal(recon_filename, recon_img)

        psnr = calculate_psnr(org_img.flatten(), recon_img.flatten())     
        bpsp = compressed_file_size * 8 / (width * height) / num_channel
        jxl_dlpr_linear["psnr"].append(psnr)
        jxl_dlpr_linear["bpsp"].append(bpsp)  
        logger.log.info(f"jxl_dlpr_linear compress level : {compress_level}, psnr : {psnr}, bpsp : {bpsp}, encode time : {encode_time:.4f}, decode time : {decode_time:.4f}")

    logger.log.info(f"***** jxl_dlpr_linear = {jxl_dlpr_linear}")

    compress_method = 'jxl-dlpr-msb'         # jxl-dlpr-linear, jxl-dlpr-msb,
    compress_levels = [1, 2, 3, 4, 5, 6, 8, 16, 32, 64, 128]     # mean tau (1~6) at jxl-dlpr-linear, jxl-dlpr-msb
    jxl_dlpr_msb = {"psnr": [], "bpsp": []}
    for compress_level in compress_levels:
        output_filename = os.path.join(output_dir, f"{compress_method}_{compress_level}.z")
        start_time = time.time()
        compressed_file_size, org_img, height, width = encode_big_file(input_filename, output_filename, compress_method, compress_level, block_size, num_channel)
        encode_time = time.time() - start_time
        start_time = time.time()
        recon_img = decode_big_file(output_filename, compress_method, block_size)
        decode_time = time.time() - start_time
        recon_filename = os.path.join(output_dir, f"{compress_method}_{compress_level}_rec.tif")
        write_tiff_with_gdal(recon_filename, recon_img)

        psnr = calculate_psnr(org_img.flatten(), recon_img.flatten())     
        bpsp = compressed_file_size * 8 / (width * height) / num_channel
        jxl_dlpr_msb["psnr"].append(psnr)
        jxl_dlpr_msb["bpsp"].append(bpsp)  
        logger.log.info(f"jxl_dlpr_msb compress level : {compress_level}, psnr : {psnr}, bpsp : {bpsp}, encode time : {encode_time:.4f}, decode time : {decode_time:.4f}")

    logger.log.info(f"***** jxl_dlpr_msb = {jxl_dlpr_msb}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        item = int(sys.argv[1])
    else:
        print("This script requires at least one argument.") 

    experiment_DL_dlpr_encode_big_file(item)
