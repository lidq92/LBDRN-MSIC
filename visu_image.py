# The file paths should be reset
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


gdal.UseExceptions()


def uint16_to_uint8(array): # with inverse Gamma correction
    min_val = array.min()
    max_val = array.max()

    scaled_array = (((array - min_val) / (max_val - min_val) ) ** (1/2.2) * 255).astype(np.uint8)

    return scaled_array


def read_RGB_NIR(input_tif):
    dataset = gdal.Open(input_tif)
    image_data = dataset.ReadAsArray()
    RGB_data = np.stack((image_data[2], image_data[1], image_data[0]), axis=-1)
    NIR_data = image_data[3]

    return RGB_data, NIR_data


def read_FCI(input_tif): # false-color image
    dataset = gdal.Open(input_tif)
    image_data = dataset.ReadAsArray()
    FCI_data = np.stack((image_data[3], image_data[2], image_data[1]), axis=-1)

    return FCI_data


def MSB_LSB():
    input_tif = "data/sample.tif" # The upper-left 2048x2048 tile of GF-2 A image 
    RGB_data, NIR_data = read_RGB_NIR(input_tif)
    R = RGB_data[:,:,0][::4,::4]
    G = RGB_data[:,:,1][::4,::4]
    B = RGB_data[:,:,2][::4,::4]
    K = 5
    MSB = (R >> K)
    LSB = R - (MSB << K)
    plt.imsave('MSB_R.png', ((MSB.astype(np.float32)/MSB.max())**(1/2.2)*255).astype(np.uint8), format='png', cmap='gray')
    plt.imsave('LSB_R.png', ((LSB.astype(np.float32)/LSB.max())**(1/2.2)*255).astype(np.uint8), format='png', cmap='gray')

    MSB = (G >> K)
    LSB = G - (MSB << K)    
    plt.imsave('MSB_G.png', ((MSB.astype(np.float32)/MSB.max())**(1/2.2)*255).astype(np.uint8), format='png', cmap='gray')
    plt.imsave('LSB_G.png', ((LSB.astype(np.float32)/LSB.max())**(1/2.2)*255).astype(np.uint8), format='png', cmap='gray')

    MSB = (B >> K)
    LSB = B - (MSB << K)
    plt.imsave('MSB_B.png', ((MSB.astype(np.float32)/MSB.max())**(1/2.2)*255).astype(np.uint8), format='png', cmap='gray')
    plt.imsave('LSB_B.png', ((LSB.astype(np.float32)/LSB.max())**(1/2.2)*255).astype(np.uint8), format='png', cmap='gray')

    MSB = (NIR_data[::4,::4] >> K)
    LSB = NIR_data[::4,::4] - (MSB << K)
    plt.imsave('MSB_N.png', ((MSB.astype(np.float32)/MSB.max())**(1/2.2)*255).astype(np.uint8), format='png', cmap='gray')
    plt.imsave('LSB_N.png', ((LSB.astype(np.float32)/LSB.max())**(1/2.2)*255).astype(np.uint8), format='png', cmap='gray')


def Gaofen2():
    files = [
        "TRIPLESAT_2_MS_L1_20191107021947_001FFCVI_002_0120200811001001_001",
        "TRIPLESAT_2_MS_L1_20191107021950_001FFCVI_003_0120200811001001_002",
        "TRIPLESAT_2_MS_L1_20191107021954_001FFCVI_004_0120200811001001_001",
        "TRIPLESAT_2_MS_L1_20200109023258_002115VI_003_0120200811001001_001",
        "TRIPLESAT_2_MS_L1_20200109023301_002115VI_004_0120200811001001_001"
    ]
    dirname = "data/GF-dataset/GF-2"
    for filename in files:
        input_tif = f"{dirname}/{filename}.tif"  
        RGB_data, NIR_data = read_RGB_NIR(input_tif)
        plt.imsave(f'{filename}_RGB.png', uint16_to_uint8(RGB_data[::14,::14]), format='png')
        plt.imsave(f'{filename}_NIR.png', uint16_to_uint8(NIR_data[::14,::14]), format='png', cmap='gray')


def Gaofen6_WFI():
    files = [        
        "GF6_WFI_Sample_A",
        "GF6_WFI_Sample_B",
        "GF6_WFI_Sample_C",
        "GF6_WFI_Sample_D",
    ]
    dirname = "data/GF-dataset/GF-6/GF6-WFI"
    for filename in files:
        input_tif = f"{dirname}/{filename}.tif"  
        RGB_data, NIR_data = read_RGB_NIR(input_tif)
        plt.imsave(f'{filename}_RGB.png', uint16_to_uint8(RGB_data[::10,::10]), format='png')
        plt.imsave(f'{filename}_NIR.png', uint16_to_uint8(NIR_data[::10,::10]), format='png', cmap='gray')


def Gaofen6_PMS():
    files = [        
        "GF6_PMS_Sample_A",
        "GF6_PMS_Sample_B",
        "GF6_PMS_Sample_C",
        "GF6_PMS_Sample_D",
    ]
    dirname = "data/GF-dataset/GF-6/GF6-PMS"
    for filename in files:
        input_tif = f"{dirname}/{filename}.tif"  
        RGB_data, NIR_data = read_RGB_NIR(input_tif)
        plt.imsave(f'{filename}_RGB.png', uint16_to_uint8(RGB_data[::10,::10]), format='png')
        plt.imsave(f'{filename}_NIR.png', uint16_to_uint8(NIR_data[::10,::10]), format='png', cmap='gray')


def error_map_low_bitrates():
    cmap = LinearSegmentedColormap.from_list('green_blue_red', ['green', 'yellow', 'red'])

    TS = 2048
    filename = "TRIPLESAT_2_MS_L1_20191107021947_001FFCVI_002_0120200811001001_001"

    input_tif = f"data/GF-dataset/GF-2/{filename}.tif"  
    org_pRGB = read_FCI(input_tif)
    pRGB8 = uint16_to_uint8(org_pRGB)
    plt.imsave(f'Aul_{TS}.png', pRGB8[:TS,:TS][::4,::4], format='png')
    plt.imsave(f'A_full.png', pRGB8[::14,::14], format='png')
    plt.imsave(f'Alr_{TS}.png', pRGB8[-TS:,-TS:][::4,::4], format='png')

    K = 8 #
    pRGB = (org_pRGB >> K) << K
    error_image_baseline = np.abs(org_pRGB.astype(np.float32) - pRGB.astype(np.float32)).mean(axis=-1)
    max_error = error_image_baseline.max()

    input_tif = f'{filename}_D16.tif' #
    pRGB = read_FCI(input_tif)
    error_image_D16 = np.abs(org_pRGB.astype(np.float32) - pRGB.astype(np.float32)).mean(axis=-1)
    max_error = max(error_image_D16.max(), max_error)

    input_tif = f'{filename}_DLPR_nll.tif' #
    pRGB = read_FCI(input_tif)
    error_image_DLPR = np.abs(org_pRGB.astype(np.float32) - pRGB.astype(np.float32)).mean(axis=-1)
    max_error = max(error_image_DLPR.max(), max_error)


    input_tif = f'outputs-rel-colors-D2/{filename}_r1_K8_bc64_nl2_D2_prec16_lr0.001_bs8192_e10/{filename}_recon.tif'
    pRGB = read_FCI(input_tif)
    error_image_LBDRN = np.abs(org_pRGB.astype(np.float32) - pRGB.astype(np.float32)).mean(axis=-1)
    max_error = max(error_image_LBDRN.max(), max_error)

    # Error map
    max_error = 255 ###
    name = 'full'
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    im0 = axs[0].imshow(error_image_baseline, cmap=cmap, vmin=0, vmax=max_error)
    axs[0].set_title('Baseline', fontsize=16)
    im1 = axs[1].imshow(error_image_D16, cmap=cmap, vmin=0, vmax=max_error)
    axs[1].set_title('BitMore (D16)', fontsize=16)
    im2 = axs[2].imshow(error_image_DLPR, cmap=cmap, vmin=0, vmax=max_error)
    axs[2].set_title('DLPR_nll', fontsize=16)
    im1 = axs[3].imshow(error_image_LBDRN, cmap=cmap, vmin=0, vmax=max_error)
    axs[3].set_title('LBDRN-MSIC', fontsize=16)
    # plt.setp([a.get_xticklabels() for a in axs], visible=False)
    # plt.setp([a.get_yticklabels() for a in axs], visible=False)
    for ax in axs:
        ax.tick_params(labelsize=14)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.86, 0.15, 0.02, 0.7])  # 
    fig.colorbar(im1, cax=cbar_ax,fraction=0.046, pad=0.04)
    plt.savefig(f'error_map_{filename}_{name}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    name = 'Aul2048'
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    im0 = axs[0].imshow(error_image_baseline[:TS,:TS], cmap=cmap, vmin=0, vmax=max_error)
    axs[0].set_title('Baseline', fontsize=16)
    im1 = axs[1].imshow(error_image_D16[:TS,:TS], cmap=cmap, vmin=0, vmax=max_error)
    axs[1].set_title('BitMore (D16)', fontsize=16)
    im2 = axs[2].imshow(error_image_DLPR[:TS,:TS], cmap=cmap, vmin=0, vmax=max_error)
    axs[2].set_title('DLPR_nll', fontsize=16)
    im1 = axs[3].imshow(error_image_LBDRN[:TS,:TS], cmap=cmap, vmin=0, vmax=max_error)
    axs[3].set_title('LBDRN-MSIC', fontsize=16)
    for ax in axs:
        ax.tick_params(labelsize=14)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.86, 0.15, 0.02, 0.7])  # 
    fig.colorbar(im1, cax=cbar_ax,fraction=0.046, pad=0.04)
    plt.savefig(f'error_map_{filename}_{name}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    name = 'Alr2048'
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    im0 = axs[0].imshow(error_image_baseline[-TS:,-TS:], cmap=cmap, vmin=0, vmax=max_error)
    axs[0].set_title('Baseline', fontsize=16)
    im1 = axs[1].imshow(error_image_D16[-TS:,-TS:], cmap=cmap, vmin=0, vmax=max_error)
    axs[1].set_title('BitMore (D16)', fontsize=16)
    im2 = axs[2].imshow(error_image_DLPR[-TS:,-TS:], cmap=cmap, vmin=0, vmax=max_error)
    axs[2].set_title('DLPR_nll', fontsize=16)
    im1 = axs[3].imshow(error_image_LBDRN[-TS:,-TS:], cmap=cmap, vmin=0, vmax=max_error)
    axs[3].set_title('LBDRN-MSIC', fontsize=16)
    for ax in axs:
        ax.tick_params(labelsize=14)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.86, 0.15, 0.02, 0.7])  # 
    fig.colorbar(im1, cax=cbar_ax,fraction=0.046, pad=0.04)
    plt.savefig(f'error_map_{filename}_{name}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    files = [
        "TRIPLESAT_2_MS_L1_20191107021947_001FFCVI_002_0120200811001001_001",
        "TRIPLESAT_2_MS_L1_20191107021950_001FFCVI_003_0120200811001001_002",
        "TRIPLESAT_2_MS_L1_20191107021954_001FFCVI_004_0120200811001001_001",
        "TRIPLESAT_2_MS_L1_20200109023258_002115VI_003_0120200811001001_001",
        "TRIPLESAT_2_MS_L1_20200109023301_002115VI_004_0120200811001001_001"
    ]
    image_IDs = ['A', 'B', 'C', 'D', 'E']
    for i, filename in enumerate(files):
        if i == 0: continue #
        input_tif = f"data/GF-dataset/GF-2/{filename}.tif"  
        org_pRGB = read_FCI(input_tif)
        pRGB8 = uint16_to_uint8(org_pRGB)

        K = 8 #
        pRGB = (org_pRGB >> K) << K
        error_image_baseline = np.abs(org_pRGB.astype(np.float32) - pRGB.astype(np.float32)).mean(axis=-1)
        max_error = error_image_baseline.max()

        input_tif = f'{filename}_D16.tif' #
        pRGB = read_FCI(input_tif)
        error_image_D16 = np.abs(org_pRGB.astype(np.float32) - pRGB.astype(np.float32)).mean(axis=-1)
        max_error = max(error_image_D16.max(), max_error)

        input_tif = f'{filename}_DLPR_nll.tif' #
        pRGB = read_FCI(input_tif)
        error_image_DLPR = np.abs(org_pRGB.astype(np.float32) - pRGB.astype(np.float32)).mean(axis=-1)
        max_error = max(error_image_DLPR.max(), max_error)

        input_tif = f'outputs-rel-colors-D2/{filename}_r1_K8_bc64_nl2_D2_prec16_lr0.001_bs8192_e10/{filename}_recon.tif'
        pRGB = read_FCI(input_tif)
        error_image_LBDRN = np.abs(org_pRGB.astype(np.float32) - pRGB.astype(np.float32)).mean(axis=-1)
        max_error = max(error_image_LBDRN.max(), max_error)

        # Error map
        max_error = 255 ###
        name = 'full'
        fig, axs = plt.subplots(1, 5, figsize=(25, 5))
        im0 = axs[0].imshow(pRGB8[::14,::14])
        axs[0].set_title(f'Image {image_IDs[i]} (NRG)', fontsize=16)
        im1 = axs[1].imshow(error_image_baseline, cmap=cmap, vmin=0, vmax=max_error)
        axs[1].set_title('Baseline', fontsize=16)
        im2 = axs[2].imshow(error_image_D16, cmap=cmap, vmin=0, vmax=max_error)
        axs[2].set_title('BitMore (D16)', fontsize=16)
        im3 = axs[3].imshow(error_image_DLPR, cmap=cmap, vmin=0, vmax=max_error)
        axs[3].set_title('DLPR_nll', fontsize=16)
        im4 = axs[4].imshow(error_image_LBDRN, cmap=cmap, vmin=0, vmax=max_error)
        axs[4].set_title('LBDRN-MSIC', fontsize=16)
        plt.setp([axs[0].get_xticklabels()], visible=False)
        plt.setp([axs[0].get_yticklabels()], visible=False)
        for ax in axs:
            ax.tick_params(labelsize=14)
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.86, 0.15, 0.02, 0.7])  # 
        fig.colorbar(im1, cax=cbar_ax,fraction=0.046, pad=0.04)
        plt.savefig(f'error_map_{filename}_{name}.png', bbox_inches='tight', pad_inches=0)
        plt.close()


def error_map_high_bitrates(c=None):
    cmap = LinearSegmentedColormap.from_list('green_blue_red', ['green', 'yellow', 'red'])

    filename = "TRIPLESAT_2_MS_L1_20191107021947_001FFCVI_002_0120200811001001_001"
    # Re-run decode.py and do not delete the reconstructed image
    # command = f'python decode.py -i outputs-rel-colors-D2/{filename}_r1_K4_bc64_nl2_D2_prec16_lr0.001_bs8192_e10/{filename}.bin -org data/GF-dataset/GF-2/{filename}.tif'
    # os.system(command)

    input_tif = f"data/GF-dataset/GF-2/{filename}.tif"  
    org_img = gdal.Open(input_tif).ReadAsArray()
    org_img = np.transpose(org_img, (1, 2, 0))

    K = 4 #
    img = (org_img >> K) << K
    error_image_baseline = np.abs(org_img.astype(np.float32) - img.astype(np.float32))
    max_error = error_image_baseline.max()

    input_tif = f'JPEG2000star.tif' #
    img = gdal.Open(input_tif).ReadAsArray()
    img = np.transpose(img, (1, 2, 0))
    error_image_JPEG2000_1 = np.abs(org_img.astype(np.float32) - img.astype(np.float32))
    max_error = max(error_image_JPEG2000_1.max(), max_error)

    input_tif = f'JPEG2000.tif' #
    img = gdal.Open(input_tif).ReadAsArray()
    img = np.transpose(img, (1, 2, 0))
    error_image_JPEG2000 = np.abs(org_img.astype(np.float32) - img.astype(np.float32))
    max_error = max(error_image_JPEG2000.max(), max_error)

    input_tif = f'JPEGXL.tif' #
    img = gdal.Open(input_tif).ReadAsArray()
    img = np.transpose(img, (1, 2, 0))
    error_image_JPEGXL = np.abs(org_img.astype(np.float32) - img.astype(np.float32))
    max_error = max(error_image_JPEGXL.max(), max_error)

    input_tif = f'ABCDedsr.tif' #
    img = gdal.Open(input_tif).ReadAsArray()
    img = np.transpose(img, (1, 2, 0))
    error_image_ABCD_edsr = np.abs(org_img.astype(np.float32) - img.astype(np.float32))
    max_error = max(error_image_ABCD_edsr.max(), max_error)

    input_tif = f'ABCDswin.tif' #
    img = gdal.Open(input_tif).ReadAsArray()
    img = np.transpose(img, (1, 2, 0))
    error_image_ABCD_swin = np.abs(org_img.astype(np.float32) - img.astype(np.float32))
    max_error = max(error_image_ABCD_swin.max(), max_error)

    input_tif = f'BitMoreD16.tif' #
    img = gdal.Open(input_tif).ReadAsArray()
    img = np.transpose(img, (1, 2, 0))
    error_image_D16 = np.abs(org_img.astype(np.float32) - img.astype(np.float32))
    max_error = max(error_image_D16.max(), max_error)

    input_tif = f'outputs-rel-colors-D2/{filename}_r1_K4_bc64_nl2_D2_prec16_lr0.001_bs8192_e10/{filename}_recon.tif'
    img = gdal.Open(input_tif).ReadAsArray()
    img = np.transpose(img, (1, 2, 0))
    error_image_LBDRN = np.abs(org_img.astype(np.float32) - img.astype(np.float32))
    max_error = max(error_image_LBDRN.max(), max_error)

    # Error map
    max_error = 16 ###
    name = 'high-bitrates'

    def plot_error_map(ax, error_image, title):
        im = ax.imshow(error_image, cmap=cmap, vmin=0, vmax=max_error)
        ax.set_title(title, fontsize=16)
        ax.tick_params(labelsize=14)
        return im
    
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    if c is None:
        error_image_baseline = error_image_baseline.mean(axis=-1)
        error_image_JPEG2000_1 = error_image_JPEG2000_1.mean(axis=-1)
        error_image_JPEG2000 = error_image_JPEG2000.mean(axis=-1)
        error_image_JPEGXL = error_image_JPEGXL.mean(axis=-1)
        error_image_D16 = error_image_D16.mean(axis=-1)
        error_image_ABCD_edsr = error_image_ABCD_edsr.mean(axis=-1)
        error_image_ABCD_swin = error_image_ABCD_swin.mean(axis=-1)
        error_image_LBDRN = error_image_LBDRN.mean(axis=-1)
    else:
        error_image_baseline = error_image_baseline[:, :, c]
        error_image_JPEG2000_1 = error_image_JPEG2000_1[:, :, c]
        error_image_JPEG2000 = error_image_JPEG2000[:, :, c]
        error_image_JPEGXL = error_image_JPEGXL[:, :, c]
        error_image_D16 = error_image_D16[:, :, c]
        error_image_ABCD_edsr = error_image_ABCD_edsr[:, :, c]
        error_image_ABCD_swin = error_image_ABCD_swin[:, :, c]
        error_image_LBDRN = error_image_LBDRN[:, :, c]

    plot_error_map(axs[0, 0], error_image_baseline, 'Baseline')
    plot_error_map(axs[0, 1], error_image_JPEG2000_1, 'JPEG 2000$^*$')
    plot_error_map(axs[0, 2], error_image_JPEG2000, 'JPEG 2000')
    plot_error_map(axs[0, 3], error_image_JPEGXL, 'JPEG XL')
    plot_error_map(axs[1, 0], error_image_D16, 'BitMore (D16)')
    plot_error_map(axs[1, 1], error_image_ABCD_edsr, 'ABCD (EDSR)')
    plot_error_map(axs[1, 2], error_image_ABCD_swin, 'ABCD (Swin)')
    im = plot_error_map(axs[1, 3], error_image_LBDRN, 'LBDRN-MSIC')

    for ax in axs:
        for sax in ax:
            sax.tick_params(labelsize=14)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.86, 0.15, 0.02, 0.7])  # 
    fig.colorbar(im, cax=cbar_ax,fraction=0.046, pad=0.04)
    save_name = f'error_map_{filename}_{name}.png'
    if c is not None: save_name = f'error_map_{filename}_{name}_{c}.png'
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == "__main__":
    MSB_LSB()
    Gaofen2()
    Gaofen6_WFI()
    Gaofen6_PMS()
    # error_map_low_bitrates()
    error_map_high_bitrates()
    error_map_high_bitrates(0)
    error_map_high_bitrates(1)
    error_map_high_bitrates(2)
    error_map_high_bitrates(3)
