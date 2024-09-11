# https://github.com/Anserw/Bjontegaard_metric
import csv
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt


def BD_PSNR(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)
    PSNR1 = np.array(PSNR1)
    PSNR2 = np.array(PSNR2)
    p1 = np.polyfit(lR1, PSNR1, 3)
    p2 = np.polyfit(lR2, PSNR2, 3)
    # integration interval
    min_int = max(min(lR1), min(lR2))
    max_int = min(max(lR1), max(lR2))
    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)
        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        # See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(lR1), PSNR1[np.argsort(lR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(lR2), PSNR2[np.argsort(lR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_diff = (int2 - int1) / (max_int - min_int)

    return avg_diff


def BD_RATE(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)
    # rate method
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)
    # integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))
    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)
        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(PSNR1), lR1[np.argsort(PSNR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(PSNR2), lR2[np.argsort(PSNR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_exp_diff = (int2 - int1) / (max_int - min_int)
    avg_diff = (np.exp(avg_exp_diff) - 1) * 100

    return avg_diff


def read_csv(csv_file_path, N=5, K=6):
    psnrs = np.zeros((N, K))
    bits_values = np.zeros((N, K))
    bpsp_values = np.zeros((N, K))
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  
        r = 0
        for row in reader:
            for I in range(N):
                psnrs[I, r] = float(row[4*I+2])  
                bits_values[I, r] = float(row[4*I+4]) 
                bpsp_values[I, r] = float(row[4*I+3]) 
            r = r + 1
            if r == K: break #

        return psnrs, bits_values, bpsp_values
    

def read_csv_lbr(csv_file_path, N=5, K=6):
    psnrs = np.zeros((N, K))
    bits_values = np.zeros((N, K))
    bpsp_values = np.zeros((N, K))
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  
        r = 0
        for row in reader:
            if r >= K - 1: 
                for I in range(N):
                    psnrs[I, r - K + 1] = float(row[4*I+2])  
                    bits_values[I, r - K + 1] = float(row[4*I+4]) 
                    bpsp_values[I, r - K + 1] = float(row[4*I+3]) 
            r = r + 1

        return psnrs, bits_values, bpsp_values
    

def feature_set():
    # Bit-depth reduction
    csv_file_path = f'SOTA_results/Baseline_11rps.csv'
    base_psnrs, base_bits_values, base_bpsp_values = read_csv(csv_file_path)

    # D = 0, 1, 2, 3
    psnrs = np.zeros((4, 5, 6))
    bits_values = np.zeros((4, 5, 6))
    bpsp_values = np.zeros((4, 5, 6))
    for D in range(4):
        if D > 0:
            csv_file_path = f'outputs-rel-colors-D{D}/results_r1_bc64_nl2_D{D}_prec16_lr0.001_bs8192_e10.csv'
        else:
            csv_file_path = f'outputs-abs-colors-D{D}/results_r1_bc64_nl2_D{D}_prec16_lr0.001_bs8192_e10.csv'
        psnrs[D], bits_values[D], bpsp_values[D] = read_csv(csv_file_path)

    for D in range(4):
        bd_psnr = BD_PSNR(base_bits_values.mean(axis=0), base_psnrs.mean(axis=0), 
                          bits_values[D].mean(axis=0), psnrs[D].mean(axis=0))
        bd_rate = BD_RATE(base_bits_values.mean(axis=0), base_psnrs.mean(axis=0), 
                          bits_values[D].mean(axis=0), psnrs[D].mean(axis=0))

    plt.figure(figsize=(8, 6)) 
    baseline_line, = plt.plot(base_bpsp_values.mean(axis=0), base_psnrs.mean(axis=0), 
            label='Baseline', color='grey', linestyle='--', marker='x', alpha=0.7)
    colors = [
        '#5C8FF9',  
        '#F6A6C9',  
        '#5AC874',  
        '#FFD580' 
    ]
    lines = [baseline_line]
    for D in [0, 1, 3, 2]:
        line, = plt.plot(bpsp_values[D].mean(axis=0), psnrs[D].mean(axis=0), 
                         color=colors[D], linestyle='-', marker='o', alpha=0.9)
        lines.append(line)
    lines[3], lines[4] = lines[4], lines[3]
    plt.legend(lines, ['Baseline', '$D$=0', '$D$=1', '$D$=2', '$D$=3'], loc='best', fontsize=16)
    plt.xlabel('Bits per sub-pixel (bpsp)', fontsize=16)
    plt.ylabel('PSNR (dB)', fontsize=16)
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16) 
    plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.4)  
    plt.savefig('RD_Curve_for_D.png', dpi=300) 
    plt.close()

    # feature set
    csv_files = ["outputs-coords/results_r1_bc64_nl2_D2_prec16_lr0.001_bs8192_e10.csv", 
                "outputs-coords-embedding/results_r1_bc64_nl2_D2_prec16_lr0.001_bs8192_e10.csv", 
                "outputs-abs-colors-D0/results_r1_bc64_nl2_D0_prec16_lr0.001_bs8192_e10.csv",
                "outputs-abs-colors-D2/results_r1_bc64_nl2_D2_prec16_lr0.001_bs8192_e10.csv",
                "outputs-rel-colors-D1/results_r1_bc64_nl2_D1_prec16_lr0.001_bs8192_e10.csv", 
                "outputs-rel-colors-D2/results_r1_bc64_nl2_D2_prec16_lr0.001_bs8192_e10.csv", 
                "outputs-rel-colors-D3/results_r1_bc64_nl2_D3_prec16_lr0.001_bs8192_e10.csv", 
                "outputs-coords-rel-colors-D2/results_r1_bc64_nl2_D2_prec16_lr0.001_bs8192_e10.csv"
                ]
    feature_sets = ["Coords", 
                    "Coords + Positional Encoding", 
                    "Abs Colors ($D=0$)",
                    "Abs Colors ($D=2$)",
                    "Rel Colors ($D=1$)",
                    "Rel Colors ($D=2$)",
                    "Rel Colors ($D=3$)",
                    "Coords + Rel Colors ($D=2$) "
                ]
    counts = [2, 50, 4, 100, 36, 100, 196, 102]
    psnrs = np.zeros((len(csv_files), 5, 6))
    bits_values = np.zeros((len(csv_files), 5, 6))
    bpsp_values = np.zeros((len(csv_files), 5, 6))
    for i, csv_file in enumerate(csv_files):
        psnrs[i], bits_values[i], bpsp_values[i] = read_csv(csv_file)

    print('\\begin{tabular}{lccc}')
    print('\t\\toprule')
    print('\tFeature Set & Count & BD-Rate & BD-PSNR \\\\')
    print('\t\\midrule')

    for i, csv_file in enumerate(csv_files):
        bd_psnr = BD_PSNR(base_bits_values.mean(axis=0), base_psnrs.mean(axis=0), 
                          bits_values[i].mean(axis=0), psnrs[i].mean(axis=0))
        bd_rate = BD_RATE(base_bits_values.mean(axis=0), base_psnrs.mean(axis=0), 
                          bits_values[i].mean(axis=0), psnrs[i].mean(axis=0))
        print(f'\t{feature_sets[i]}  & {counts[i]} & ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ \\\\')
        if i in [1,6]:
            print('\t\\midrule')
    print('\t\\bottomrule')
    print('\\end{tabular}')


def network_hyperparameter():
    # Bit-depth reduction
    csv_file_path = f'SOTA_results/Baseline_11rps.csv'
    base_psnrs, base_bits_values, base_bpsp_values = read_csv(csv_file_path)

    # (F, L)
    FL = [(128, 1), (128, 2), (256, 2), (64, 2)] # 
    psnrs = np.zeros((len(FL), 5, 6))
    bits_values = np.zeros((len(FL), 5, 6))
    bpsp_values = np.zeros((len(FL), 5, 6))
    for i, (F, L) in enumerate(FL):
        csv_file_path = f'outputs-rel-colors-D2/results_r1_bc{F}_nl{L}_D2_prec16_lr0.001_bs8192_e10.csv'
        psnrs[i], bits_values[i], bpsp_values[i] = read_csv(csv_file_path)


    for i, (F, L) in enumerate(FL):
        bd_psnr = BD_PSNR(base_bits_values.mean(axis=0), base_psnrs.mean(axis=0), 
                          bits_values[i].mean(axis=0), psnrs[i].mean(axis=0))
        bd_rate = BD_RATE(base_bits_values.mean(axis=0), base_psnrs.mean(axis=0), 
                          bits_values[i].mean(axis=0), psnrs[i].mean(axis=0))
    
    parameters = ['13,444','29,956','92,676','10,884']
    print('\\begin{tabular}{lccc}')
    print('\t\\toprule')
    print('\t$(F,L)$ & \\# Parameters & BD-Rate & BD-PSNR \\\\')
    print('\t\\midrule')

    for i, (F, L) in enumerate(FL):
        bd_psnr = BD_PSNR(base_bits_values.mean(axis=0), base_psnrs.mean(axis=0), 
                          bits_values[i].mean(axis=0), psnrs[i].mean(axis=0))
        bd_rate = BD_RATE(base_bits_values.mean(axis=0), base_psnrs.mean(axis=0), 
                          bits_values[i].mean(axis=0), psnrs[i].mean(axis=0))
        print(f'\t$({F},{L})$  & {parameters[i]} & ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ \\\\')
        if i in [1,6]:
            print('\t\\midrule')
    print('\t\\bottomrule')
    print('\\end{tabular}')


def training_hyperparameter():
    # Bit-depth reduction
    csv_file_path = f'SOTA_results/Baseline_11rps.csv'
    base_psnrs, base_bits_values, base_bpsp_values = read_csv(csv_file_path)

    # lr
    lrs = [0.01, 0.001, 0.0001]
    psnrs = np.zeros((len(lrs), 5, 6))
    bits_values = np.zeros((len(lrs), 5, 6))
    bpsp_values = np.zeros((len(lrs), 5, 6))
    for i, lr in enumerate(lrs):
        csv_file_path = f'outputs-rel-colors-D2/results_r1_bc64_nl2_D2_prec16_lr{lr}_bs8192_e10.csv'
        psnrs[i], bits_values[i], bpsp_values[i] = read_csv(csv_file_path)

    bd_psnrs = np.zeros(3)
    bd_rates = np.zeros(3)
    for i, lr in enumerate(lrs):
        bd_psnrs[i] = BD_PSNR(base_bits_values.mean(axis=0), base_psnrs.mean(axis=0), 
                          bits_values[i].mean(axis=0), psnrs[i].mean(axis=0))
        bd_rates[i] = BD_RATE(base_bits_values.mean(axis=0), base_psnrs.mean(axis=0), 
                          bits_values[i].mean(axis=0), psnrs[i].mean(axis=0))
    
    plt.figure(figsize=(8, 6)) 
    fig, ax1 = plt.subplots()
    bar_width = 0.35
    colors = [
        '#5C8FF9',  
        '#F6A6C9',  
        '#5AC874',  
        '#FFD580' 
    ]
    ax1.bar([i-bar_width/2-0.01 for i in range(len(lrs))], bd_psnrs, color=colors[0], label='BD-PSNR', width=bar_width, alpha=0.99)
    ax1.set_xlabel('Learning Rate', fontsize=16)
    ax1.set_ylabel('BD-PSNR (dB)', color='blue', fontsize=16)
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=16)

    ax2 = ax1.twinx()
    ax2.bar([i+bar_width/2+0.01 for i in range(len(lrs))], bd_rates, color=colors[1], label='BD-Rate', width=bar_width, alpha=0.99)
    ax2.set_ylabel('BD-Rate', color='red', fontsize=16)
    ax2.tick_params(axis='y', labelcolor='red')

    max_bd_rate = max(bd_rates)
    min_bd_rate = min(bd_rates)
    ax2.set_ylim(min_bd_rate - 1, max_bd_rate + 1)  
    ax2.invert_yaxis()
    ax2.set_yticks([-16,-17,-18,-19, -20], [-16,-17,-18,-19, -20], fontsize=16)
    ax2.set_yticklabels(['-16%','-17%','-18%','-19%','-20%'])

    ax1.set_xticks(range(len(lrs)), lrs, fontsize=16)
    ax1.set_xticklabels(lrs)

    plt.tight_layout()
    plt.savefig(f'BD-PSNR-lr.png', dpi=300) 
    plt.close()

    # batch sizes
    bss = [2048, 4096, 8192]
    psnrs = np.zeros((len(bss), 5, 6))
    bits_values = np.zeros((len(bss), 5, 6))
    bpsp_values = np.zeros((len(bss), 5, 6))
    for i, bs in enumerate(bss):
        csv_file_path = f'outputs-rel-colors-D2/results_r1_bc64_nl2_D2_prec16_lr0.001_bs{bs}_e10.csv'
        psnrs[i], bits_values[i], bpsp_values[i] = read_csv(csv_file_path)
    bd_psnrs = np.zeros(3)
    bd_rates = np.zeros(3)
    print('\\begin{tabular}{lcc}')
    print('\t\\toprule')
    print('\tBatch Size & BD-Rate & BD-PSNR \\\\')
    print('\t\\midrule')
    for i, bs in enumerate(bss):
        bd_psnrs[i] = BD_PSNR(base_bits_values.mean(axis=0), base_psnrs.mean(axis=0), 
                          bits_values[i].mean(axis=0), psnrs[i].mean(axis=0))
        bd_rates[i] = BD_RATE(base_bits_values.mean(axis=0), base_psnrs.mean(axis=0), 
                          bits_values[i].mean(axis=0), psnrs[i].mean(axis=0))
        print(f'\t{bs} & ${round(bd_rates[i],3):.3f}\%$ & ${round(bd_psnrs[i],3):.3f}$ \\\\')
    print('\t\\bottomrule')
    print('\\end{tabular}')
        
    # epochs
    epochs = [1, 5, 10, 15]
    psnrs = np.zeros((len(epochs), 5, 6))
    bits_values = np.zeros((len(epochs), 5, 6))
    bpsp_values = np.zeros((len(epochs), 5, 6))
    for i, e in enumerate(epochs):
        csv_file_path = f'outputs-rel-colors-D2/results_r1_bc64_nl2_D2_prec16_lr0.001_bs8192_e{e}.csv'
        psnrs[i], bits_values[i], bpsp_values[i] = read_csv(csv_file_path)
    bd_psnrs = np.zeros(len(epochs))
    bd_rates = np.zeros(len(epochs))
    for i, e in enumerate(epochs):
        bd_psnrs[i] = BD_PSNR(base_bits_values.mean(axis=0), base_psnrs.mean(axis=0), 
                          bits_values[i].mean(axis=0), psnrs[i].mean(axis=0))
        bd_rates[i] = BD_RATE(base_bits_values.mean(axis=0), base_psnrs.mean(axis=0), 
                          bits_values[i].mean(axis=0), psnrs[i].mean(axis=0))

    plt.figure(figsize=(8, 6))
    fig, ax1 = plt.subplots()
    ax1.plot(epochs, bd_psnrs, marker='o', color='blue', label='BD-PSNR')
    ax1.set_xlabel('Number of Epochs', fontsize=16)
    ax1.set_ylabel('BD-PSNR (dB)', fontsize=16, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=16)
    ax2 = ax1.twinx()
    ax2.plot(epochs, bd_rates, marker='s', color='red', label='BD-Rate')
    ax2.set_ylabel('BD-Rate', fontsize=16, color='red')
    ax2.tick_params(axis='y', labelcolor='red', labelsize=16)
    max_bd_rate = max(bd_rates)
    min_bd_rate = min(bd_rates)
    ax2.set_ylim(min_bd_rate - 0.1, max_bd_rate + 0.1)  
    ax2.invert_yaxis()
    ax2.set_yticks([-19.0,-19.1,-19.2,-19.3, -19.4, -19.5, -19.6], 
                   [-19.0,-19.1,-19.2,-19.3, -19.4, -19.5, -19.6], fontsize=16)
    ax2.set_yticklabels(['-19.0%','-19.1%','-19.2%','-19.3%', '-19.4%', '-19.5%', '-19.6%'])
    ax1.set_xticks(epochs)
    ax1.set_xticklabels(['{}'.format(e) for e in epochs], fontsize=16)

    plt.tight_layout()
    plt.savefig('BD-PSNR-e.png', dpi=300)
    plt.close()


def SOTA():
    # Bit-depth reduction
    csv_file_path = f'SOTA_results/Baseline_11rps.csv'
    base_psnrs, base_bits_values, base_bpsp_values = read_csv(csv_file_path, N=13)

    csv_file_path = f'outputs-rel-colors-D2/results_r1_bc64_nl2_D2_prec16_lr0.001_bs8192_e10.csv'
    LBDRN_psnrs, LBDRN_bits_values, LBDRN_bpsp_values = read_csv(csv_file_path, N=13)

    csv_file_path = f'SOTA_results/JPEG2000_11rps.csv'
    jpeg2000_psnrs, jpeg2000_bits_values, jpeg2000_bpsp_values = read_csv(csv_file_path, N=13)

    csv_file_path = f'SOTA_results/JPEG2000star_11rps.csv'
    jpeg2000star_psnrs, jpeg2000star_bits_values, jpeg2000star_bpsp_values = read_csv(csv_file_path, N=13)

    csv_file_path = f'SOTA_results/JPEGXL_11rps.csv'
    jpegxl_psnrs, jpegxl_bits_values, jpegxl_bpsp_values = read_csv(csv_file_path, N=13)

    methods = ['bitmore', 'ABCD_edsr', 'ABCD_swin']
    bdr_psnrs = np.zeros((len(methods), 13, 6))
    for i, method in enumerate(methods):
        csv_file_path = f'SOTA_results/test_{method}.csv'
        with open(csv_file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  
            r = 0
            for row in reader:
                if r > 5 and r < 12:
                    for I in range(5):
                        bdr_psnrs[i, I, 11-r] = float(row[I+1]) #
                r = r + 1
        csv_file_path = f'SOTA_results/test_{method}_GF6.csv'
        with open(csv_file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  
            r = 0
            for row in reader:
                if r > 4 and r < 11:
                    for I in range(8): # 4 PMS + 4 WFI, not 4 WFI + 4PMS order
                        bdr_psnrs[i, I+5, 10-r] = float(row[I+1]) #
                r = r + 1

    mean_bd_rates = np.zeros((7, 13))
    mean_bd_psnrs = np.zeros((7, 13))
    midx = 0
    print('\\begin{tabular}{lcccccccccc}')
    print('\t\\toprule')
    print('\tAgainst & \multicolumn{2}{c}{GF-2 A} & \multicolumn{2}{c}{GF-2 B} & \multicolumn{2}{c}{GF-2 C} & \multicolumn{2}{c}{GF-2 D} & \multicolumn{2}{c}{GF-2 E} \\\\')
    print('\t Method & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR \\\\')
    print('\t\\midrule')
    s = 'Baseline '
    for I in range(5): # GF-2
        bd_rate = BD_RATE(base_bits_values[I], base_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], base_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    print('\t\\midrule')
    s = 'JPEG 2000$^*$ '
    for I in range(5): 
        bd_rate = BD_RATE(jpeg2000star_bits_values[I], jpeg2000star_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(jpeg2000star_bits_values[I], jpeg2000star_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')

    s = 'JPEG 2000 '
    for I in range(5):
        bd_rate = BD_RATE(jpeg2000_bits_values[I], jpeg2000_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(jpeg2000_bits_values[I], jpeg2000_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    s = 'JPEG XL '
    for I in range(5):
        bd_rate = BD_RATE(jpegxl_bits_values[I],jpegxl_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(jpegxl_bits_values[I], jpegxl_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')    
    print('\t\\midrule')
    s = 'BitMore (D16) '
    for I in range(5):
        bd_rate = BD_RATE(base_bits_values[I], bdr_psnrs[0][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], bdr_psnrs[0][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    s = 'ABCD (EDSR) '
    for I in range(5):
        bd_rate = BD_RATE(base_bits_values[I], bdr_psnrs[1][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], bdr_psnrs[1][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    s = 'ABCD (Swin) '
    for I in range(5):
        bd_rate = BD_RATE(base_bits_values[I], bdr_psnrs[2][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], bdr_psnrs[2][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    print('\t\\bottomrule')
    print('\\end{tabular}')

    midx = 0
    print('\n\\vspace{2mm}\\begin{tabular}{lcccccccc}')
    print('\t\\toprule')
    print('\tAgainst & \multicolumn{2}{c}{GF-6 WFI A} & \multicolumn{2}{c}{GF-6 WFI B} & \multicolumn{2}{c}{GF-6 WFI C} & \multicolumn{2}{c}{GF-6 WFI D} \\\\')
    print('\t Method & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR \\\\')
    print('\t\\midrule')
    s = 'Baseline '
    for I in range(5, 9): # GF-6 WFI
        bd_rate = BD_RATE(base_bits_values[I], base_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], base_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    print('\t\\midrule')
    s = 'JPEG 2000$^*$ '
    for I in range(5, 9):
        bd_rate = BD_RATE(jpeg2000star_bits_values[I], jpeg2000star_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(jpeg2000star_bits_values[I], jpeg2000star_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')

    s = 'JPEG 2000 '
    for I in range(5, 9):
        bd_rate = BD_RATE(jpeg2000_bits_values[I], jpeg2000_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(jpeg2000_bits_values[I], jpeg2000_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    s = 'JPEG XL '
    for I in range(5, 9):
        bd_rate = BD_RATE(jpegxl_bits_values[I],jpegxl_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(jpegxl_bits_values[I], jpegxl_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')    
    print('\t\\midrule')
    s = 'BitMore (D16) '
    for I in range(5, 9):
        bd_rate = BD_RATE(base_bits_values[I], bdr_psnrs[0][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], bdr_psnrs[0][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    s = 'ABCD (EDSR) '
    for I in range(5, 9):
        bd_rate = BD_RATE(base_bits_values[I], bdr_psnrs[1][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], bdr_psnrs[1][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    s = 'ABCD (Swin) '
    for I in range(5, 9):
        bd_rate = BD_RATE(base_bits_values[I], bdr_psnrs[2][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], bdr_psnrs[2][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    print('\t\\bottomrule')
    print('\\end{tabular}')

    midx = 0
    print('\n\\vspace{2mm}\\begin{tabular}{lcccccccc}')
    print('\t\\toprule')
    print('\tAgainst & \multicolumn{2}{c}{GF-6 PMS A} & \multicolumn{2}{c}{GF-6 PMS B} & \multicolumn{2}{c}{GF-6 PMS C} & \multicolumn{2}{c}{GF-6 PMS D} \\\\')
    print('\t Method & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR \\\\')
    print('\t\\midrule')
    s = 'Baseline '
    for I in range(9, 13): # GF-6 PMS
        bd_rate = BD_RATE(base_bits_values[I], base_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], base_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    print('\t\\midrule')
    s = 'JPEG 2000$^*$ '
    for I in range(9, 13): 
        bd_rate = BD_RATE(jpeg2000star_bits_values[I], jpeg2000star_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(jpeg2000star_bits_values[I], jpeg2000star_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')

    s = 'JPEG 2000 '
    for I in range(9, 13):
        bd_rate = BD_RATE(jpeg2000_bits_values[I], jpeg2000_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(jpeg2000_bits_values[I], jpeg2000_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    s = 'JPEG XL '
    for I in range(9, 13):
        bd_rate = BD_RATE(jpegxl_bits_values[I],jpegxl_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(jpegxl_bits_values[I], jpegxl_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')    
    print('\t\\midrule')
    s = 'BitMore (D16) '
    for I in range(9, 13):
        bd_rate = BD_RATE(base_bits_values[I], bdr_psnrs[0][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], bdr_psnrs[0][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    s = 'ABCD (EDSR) '
    for I in range(9, 13):
        bd_rate = BD_RATE(base_bits_values[I], bdr_psnrs[1][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], bdr_psnrs[1][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    s = 'ABCD (Swin) '
    for I in range(9, 13):
        bd_rate = BD_RATE(base_bits_values[I], bdr_psnrs[2][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], bdr_psnrs[2][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    print('\t\\bottomrule')
    print('\\end{tabular}')

    print(f'Average performance: {mean_bd_rates.mean(axis=1)}, {mean_bd_psnrs.mean(axis=1)}')

    for I in range(13):
        plt.figure(figsize=(8, 6)) 
        plt.plot(base_bpsp_values[I], base_psnrs[I], 
                 label='Baseline', color='grey', linestyle='--', marker='x', alpha=0.6)
        plt.plot(jpeg2000star_bpsp_values[I], jpeg2000star_psnrs[I], 
                 label='JPEG 2000$^*$', linestyle='-', marker='o', alpha=0.9)
        plt.plot(jpeg2000_bpsp_values[I], jpeg2000_psnrs[I], 
                 label='JPEG 2000', linestyle='-', marker='o', alpha=0.9)
        plt.plot(jpegxl_bpsp_values[I], jpegxl_psnrs[I], 
                 label='JPEG XL', linestyle='-', marker='o', alpha=0.9)
        plt.plot(base_bpsp_values[I], bdr_psnrs[0][I], 
                 label='BitMore (D16)', linestyle='-', marker='o', alpha=0.9)
        plt.plot(base_bpsp_values[I], bdr_psnrs[1][I], 
                 label='ABCD (EDSR)', linestyle='-', marker='o', alpha=0.9)
        plt.plot(base_bpsp_values[I], bdr_psnrs[2][I], 
                 label='ABCD (Swin)', linestyle='-', marker='o', alpha=0.9)
        plt.plot(LBDRN_bpsp_values[I], LBDRN_psnrs[I], 
                 label='LBDRN-MSIC', linestyle='-', marker='o', alpha=0.9)
        plt.legend(loc='best')
        plt.xlabel('Bits per sub-pixel (bpsp)', fontsize=16)
        plt.ylabel('PSNR (dB)', fontsize=16)
        plt.xticks(fontsize=16) 
        plt.yticks(fontsize=16) 
        plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.4)  
        plt.savefig(f'RD_Curve_SOTA_{I}.png', dpi=300) 
        plt.close()

    plt.figure(figsize=(8, 6)) 
    plt.plot(base_bpsp_values[:5].mean(axis=0), base_psnrs[:5].mean(axis=0), 
                label='Baseline', color='grey', linestyle='--', marker='x', alpha=0.6)
    plt.plot(jpeg2000star_bpsp_values[:5].mean(axis=0), jpeg2000star_psnrs[:5].mean(axis=0), 
                label='JPEG 2000$^*$', linestyle='-', marker='o', alpha=0.9)
    plt.plot(jpeg2000_bpsp_values[:5].mean(axis=0), jpeg2000_psnrs[:5].mean(axis=0), 
                label='JPEG 2000', linestyle='-', marker='o', alpha=0.9)
    plt.plot(jpegxl_bpsp_values[:5].mean(axis=0), jpegxl_psnrs[:5].mean(axis=0), 
                label='JPEG XL', linestyle='-', marker='o', alpha=0.9)
    plt.plot(base_bpsp_values[:5].mean(axis=0), bdr_psnrs[0][:5].mean(axis=0), 
                label='BitMore (D16)', linestyle='-', marker='o', alpha=0.9)
    plt.plot(base_bpsp_values[:5].mean(axis=0), bdr_psnrs[1][:5].mean(axis=0), 
                label='ABCD (EDSR)', linestyle='-', marker='o', alpha=0.9)
    plt.plot(base_bpsp_values[:5].mean(axis=0), bdr_psnrs[2][:5].mean(axis=0), 
                label='ABCD (Swin)', linestyle='-', marker='o', alpha=0.9)
    plt.plot(LBDRN_bpsp_values[:5].mean(axis=0), LBDRN_psnrs[:5].mean(axis=0), 
                label='LBDRN-MSIC', linestyle='-', marker='o', alpha=0.9)
    plt.legend(loc='best')
    plt.xlabel('Bits per sub-pixel (bpsp)', fontsize=16)
    plt.ylabel('PSNR (dB)', fontsize=16)
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16) 
    plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.4)  
    plt.savefig(f'Average_RD_Curve_SOTA.png', dpi=300) 
    plt.close()

    plt.figure(figsize=(8, 6)) 
    plt.plot(base_bpsp_values[5:9].mean(axis=0), base_psnrs[5:9].mean(axis=0), 
                label='Baseline', color='grey', linestyle='--', marker='x', alpha=0.6)
    plt.plot(jpeg2000star_bpsp_values[5:9].mean(axis=0), jpeg2000star_psnrs[5:9].mean(axis=0), 
                label='JPEG 2000$^*$', linestyle='-', marker='o', alpha=0.9)
    plt.plot(jpeg2000_bpsp_values[5:9].mean(axis=0), jpeg2000_psnrs[5:9].mean(axis=0), 
                label='JPEG 2000', linestyle='-', marker='o', alpha=0.9)
    plt.plot(jpegxl_bpsp_values[5:9].mean(axis=0), jpegxl_psnrs[5:9].mean(axis=0), 
                label='JPEG XL', linestyle='-', marker='o', alpha=0.9)
    plt.plot(base_bpsp_values[5:9].mean(axis=0), bdr_psnrs[0][5:9].mean(axis=0), 
                label='BitMore (D16)', linestyle='-', marker='o', alpha=0.9)
    plt.plot(base_bpsp_values[5:9].mean(axis=0), bdr_psnrs[1][5:9].mean(axis=0), 
                label='ABCD (EDSR)', linestyle='-', marker='o', alpha=0.9)
    plt.plot(base_bpsp_values[5:9].mean(axis=0), bdr_psnrs[2][5:9].mean(axis=0), 
                label='ABCD (Swin)', linestyle='-', marker='o', alpha=0.9)
    plt.plot(LBDRN_bpsp_values[5:9].mean(axis=0), LBDRN_psnrs[5:9].mean(axis=0), 
                label='LBDRN-MSIC', linestyle='-', marker='o', alpha=0.9)
    plt.legend(loc='best')
    plt.xlabel('Bits per sub-pixel (bpsp)', fontsize=16)
    plt.ylabel('PSNR (dB)', fontsize=16)
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16) 
    plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.4)  
    plt.savefig(f'Average_RD_Curve_GF6_WFI_SOTA.png', dpi=300) 
    plt.close()

    plt.figure(figsize=(8, 6)) 
    plt.plot(base_bpsp_values[9:13].mean(axis=0), base_psnrs[9:13].mean(axis=0), 
                label='Baseline', color='grey', linestyle='--', marker='x', alpha=0.6)
    plt.plot(jpeg2000star_bpsp_values[9:13].mean(axis=0), jpeg2000star_psnrs[9:13].mean(axis=0), 
                label='JPEG 2000$^*$', linestyle='-', marker='o', alpha=0.9)
    plt.plot(jpeg2000_bpsp_values[9:13].mean(axis=0), jpeg2000_psnrs[9:13].mean(axis=0), 
                label='JPEG 2000', linestyle='-', marker='o', alpha=0.9)
    plt.plot(jpegxl_bpsp_values[9:13].mean(axis=0), jpegxl_psnrs[9:13].mean(axis=0), 
                label='JPEG XL', linestyle='-', marker='o', alpha=0.9)
    plt.plot(base_bpsp_values[9:13].mean(axis=0), bdr_psnrs[0][9:13].mean(axis=0), 
                label='BitMore (D16)', linestyle='-', marker='o', alpha=0.9)
    plt.plot(base_bpsp_values[9:13].mean(axis=0), bdr_psnrs[1][9:13].mean(axis=0), 
                label='ABCD (EDSR)', linestyle='-', marker='o', alpha=0.9)
    plt.plot(base_bpsp_values[9:13].mean(axis=0), bdr_psnrs[2][9:13].mean(axis=0), 
                label='ABCD (Swin)', linestyle='-', marker='o', alpha=0.9)
    plt.plot(LBDRN_bpsp_values[9:13].mean(axis=0), LBDRN_psnrs[9:13].mean(axis=0), 
                label='LBDRN-MSIC', linestyle='-', marker='o', alpha=0.9)
    plt.legend(loc='best')
    plt.xlabel('Bits per sub-pixel (bpsp)', fontsize=16)
    plt.ylabel('PSNR (dB)', fontsize=16)
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16) 
    plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.4)  
    plt.savefig(f'Average_RD_Curve_GF6_PMS_SOTA.png', dpi=300) 
    plt.close()


def SOTA_lowbitrates():
    # Bit-depth reduction
    csv_file_path = f'SOTA_results/Baseline_11rps.csv'
    base_psnrs, base_bits_values, base_bpsp_values = read_csv_lbr(csv_file_path, N=13)

    csv_file_path = f'outputs-rel-colors-D2/results_r1_bc64_nl2_D2_prec16_lr0.001_bs8192_e10.csv'
    LBDRN_psnrs, LBDRN_bits_values, LBDRN_bpsp_values = read_csv_lbr(csv_file_path, N=13)

    csv_file_path = f'SOTA_results/JPEG2000_11rps.csv'
    jpeg2000_psnrs, jpeg2000_bits_values, jpeg2000_bpsp_values = read_csv_lbr(csv_file_path, N=13)

    csv_file_path = f'SOTA_results/JPEGXL_11rps.csv'
    jpegxl_psnrs, jpegxl_bits_values, jpegxl_bpsp_values = read_csv_lbr(csv_file_path, N=13)

    csv_file_path = f'SOTA_results/DLPR_nll_11rps.csv'
    dlpr_psnrs, dlpr_bits_values, dlpr_bpsp_values = read_csv_lbr(csv_file_path, N=13)


    methods = ['bitmore', 'ABCD_edsr', 'ABCD_swin']
    bdr_psnrs = np.zeros((len(methods), 13, 6))
    for i, method in enumerate(methods):
        csv_file_path = f'SOTA_results/test_{method}.csv'
        with open(csv_file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  
            r = 0
            for row in reader:
                if r > 0 and r < 7:
                    for I in range(5):
                        bdr_psnrs[i, I, 6-r] = float(row[I+1]) #
                r = r + 1
        csv_file_path = f'SOTA_results/test_{method}_GF6.csv'
        with open(csv_file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  
            r = 0
            for row in reader:
                if r < 6:
                    for I in range(8): # 4 PMS + 4 WFI, not 4 WFI + 4PMS order
                        bdr_psnrs[i, I+5, 5-r] = float(row[I+1]) #
                r = r + 1

    mean_bd_rates = np.zeros((7, 13))
    mean_bd_psnrs = np.zeros((7, 13))
    midx = 0
    print('\\begin{tabular}{lcccccccccc}')
    print('\t\\toprule')
    print('\tAgainst & \multicolumn{2}{c}{GF-2 A} & \multicolumn{2}{c}{GF-2 B} & \multicolumn{2}{c}{GF-2 C} & \multicolumn{2}{c}{GF-2 D} & \multicolumn{2}{c}{GF-2 E} \\\\')
    print('\t Method & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR \\\\')
    print('\t\\midrule')
    s = 'Baseline '
    for I in range(5): # GF-2
        bd_rate = BD_RATE(base_bits_values[I], base_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], base_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    print('\t\\midrule')

    s = 'JPEG 2000 '
    for I in range(5):
        bd_rate = BD_RATE(jpeg2000_bits_values[I], jpeg2000_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(jpeg2000_bits_values[I], jpeg2000_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    s = 'JPEG XL '
    for I in range(5):
        bd_rate = BD_RATE(jpegxl_bits_values[I],jpegxl_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(jpegxl_bits_values[I], jpegxl_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')  

    s = 'DLPR\_nll '
    for I in range(5): 
        bd_rate = BD_RATE(dlpr_bits_values[I], dlpr_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(dlpr_bits_values[I], dlpr_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')  

    print('\t\\midrule')
    s = 'BitMore (D16) '
    for I in range(5):
        bd_rate = BD_RATE(base_bits_values[I], bdr_psnrs[0][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], bdr_psnrs[0][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    s = 'ABCD (EDSR) '
    for I in range(5):
        bd_rate = BD_RATE(base_bits_values[I], bdr_psnrs[1][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], bdr_psnrs[1][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    s = 'ABCD (Swin) '
    for I in range(5):
        bd_rate = BD_RATE(base_bits_values[I], bdr_psnrs[2][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], bdr_psnrs[2][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    print('\t\\bottomrule')
    print('\\end{tabular}')

    midx = 0
    print('\n\\vspace{2mm}\\begin{tabular}{lcccccccc}')
    print('\t\\toprule')
    print('\tAgainst & \multicolumn{2}{c}{GF-6 WFI A} & \multicolumn{2}{c}{GF-6 WFI B} & \multicolumn{2}{c}{GF-6 WFI C} & \multicolumn{2}{c}{GF-6 WFI D} \\\\')
    print('\t Method & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR \\\\')
    print('\t\\midrule')
    s = 'Baseline '
    for I in range(5, 9): # GF-6 WFI
        bd_rate = BD_RATE(base_bits_values[I], base_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], base_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    print('\t\\midrule')

    s = 'JPEG 2000 '
    for I in range(5, 9):
        bd_rate = BD_RATE(jpeg2000_bits_values[I], jpeg2000_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(jpeg2000_bits_values[I], jpeg2000_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    s = 'JPEG XL '
    for I in range(5, 9):
        bd_rate = BD_RATE(jpegxl_bits_values[I],jpegxl_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(jpegxl_bits_values[I], jpegxl_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')    
    s = 'DLPR\_nll '
    for I in range(5, 9): 
        bd_rate = BD_RATE(dlpr_bits_values[I], dlpr_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(dlpr_bits_values[I], dlpr_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')  

    print('\t\\midrule')
    s = 'BitMore (D16) '
    for I in range(5, 9):
        bd_rate = BD_RATE(base_bits_values[I], bdr_psnrs[0][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], bdr_psnrs[0][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    s = 'ABCD (EDSR) '
    for I in range(5, 9):
        bd_rate = BD_RATE(base_bits_values[I], bdr_psnrs[1][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], bdr_psnrs[1][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    s = 'ABCD (Swin) '
    for I in range(5, 9):
        bd_rate = BD_RATE(base_bits_values[I], bdr_psnrs[2][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], bdr_psnrs[2][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    print('\t\\bottomrule')
    print('\\end{tabular}')

    midx = 0
    print('\n\\vspace{2mm}\\begin{tabular}{lcccccccc}')
    print('\t\\toprule')
    print('\tAgainst & \multicolumn{2}{c}{GF-6 PMS A} & \multicolumn{2}{c}{GF-6 PMS B} & \multicolumn{2}{c}{GF-6 PMS C} & \multicolumn{2}{c}{GF-6 PMS D} \\\\')
    print('\t Method & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR \\\\')
    print('\t\\midrule')
    s = 'Baseline '
    for I in range(9, 13): # GF-6 PMS
        bd_rate = BD_RATE(base_bits_values[I], base_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], base_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')

    print('\t\\midrule')

    s = 'JPEG 2000 '
    for I in range(9, 13):
        bd_rate = BD_RATE(jpeg2000_bits_values[I], jpeg2000_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(jpeg2000_bits_values[I], jpeg2000_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    s = 'JPEG XL '
    for I in range(9, 13):
        bd_rate = BD_RATE(jpegxl_bits_values[I],jpegxl_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(jpegxl_bits_values[I], jpegxl_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')   

    s = 'DLPR\_nll '
    for I in range(9, 13): 
        bd_rate = BD_RATE(dlpr_bits_values[I], dlpr_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(dlpr_bits_values[I], dlpr_psnrs[I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')  

    print('\t\\midrule')
    s = 'BitMore (D16) '
    for I in range(9, 13):
        bd_rate = BD_RATE(base_bits_values[I], bdr_psnrs[0][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], bdr_psnrs[0][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    s = 'ABCD (EDSR) '
    for I in range(9, 13):
        bd_rate = BD_RATE(base_bits_values[I], bdr_psnrs[1][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], bdr_psnrs[1][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    s = 'ABCD (Swin) '
    for I in range(9, 13):
        bd_rate = BD_RATE(base_bits_values[I], bdr_psnrs[2][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        bd_psnr = BD_PSNR(base_bits_values[I], bdr_psnrs[2][I],
                          LBDRN_bits_values[I], LBDRN_psnrs[I])
        s += f'& ${round(bd_rate,3):.3f}\%$ & ${round(bd_psnr,3):.3f}$ '
        mean_bd_rates[midx, I] += round(bd_rate,3)
        mean_bd_psnrs[midx, I] += round(bd_psnr,3)
    midx += 1
    print(s + '\\\\')
    print('\t\\bottomrule')
    print('\\end{tabular}')

    print(f'Average performance: {mean_bd_rates.mean(axis=1)}, {mean_bd_psnrs.mean(axis=1)}')

    midx = 0
    print('\\begin{tabular}{lcccccc}')
    print('\t\\toprule')
    print('\tAgainst & \multicolumn{2}{c}{GF-2} & \multicolumn{2}{c}{GF-6 PMS} & \multicolumn{2}{c}{GF-6 WFI} \\\\')
    print('\t Method & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR & BD-Rate & BD-PSNR \\\\')
    print('\t\\midrule')
    s = 'Baseline '
    s += f'& ${mean_bd_rates[midx, :5].mean():.3f}\%$ & ${mean_bd_psnrs[midx, :5].mean():.3f}$ '
    s += f'& ${mean_bd_rates[midx, 9:13].mean():.3f}\%$ & ${mean_bd_psnrs[midx, 9:13].mean():.3f}$ '
    s += f'& ${mean_bd_rates[midx, 5:9].mean():.3f}\%$ & ${mean_bd_psnrs[midx, 5:9].mean():.3f}$ '
    midx += 1
    print(s + '\\\\')
    print('\t\\midrule')

    s = 'JPEG 2000 '
    s += f'& ${mean_bd_rates[midx, :5].mean():.3f}\%$ & ${mean_bd_psnrs[midx, :5].mean():.3f}$ '
    s += f'& ${mean_bd_rates[midx, 9:13].mean():.3f}\%$ & ${mean_bd_psnrs[midx, 9:13].mean():.3f}$ '
    s += f'& ${mean_bd_rates[midx, 5:9].mean():.3f}\%$ & ${mean_bd_psnrs[midx, 5:9].mean():.3f}$ '
    midx += 1
    print(s + '\\\\')
    
    s = 'JPEG XL '
    s += f'& ${mean_bd_rates[midx, :5].mean():.3f}\%$ & ${mean_bd_psnrs[midx, :5].mean():.3f}$ '
    s += f'& ${mean_bd_rates[midx, 9:13].mean():.3f}\%$ & ${mean_bd_psnrs[midx, 9:13].mean():.3f}$ '
    s += f'& ${mean_bd_rates[midx, 5:9].mean():.3f}\%$ & ${mean_bd_psnrs[midx, 5:9].mean():.3f}$ '
    midx += 1
    print(s + '\\\\')  

    s = 'DLPR\_nll '
    s += f'& ${mean_bd_rates[midx, :5].mean():.3f}\%$ & ${mean_bd_psnrs[midx, :5].mean():.3f}$ '
    s += f'& ${mean_bd_rates[midx, 9:13].mean():.3f}\%$ & ${mean_bd_psnrs[midx, 9:13].mean():.3f}$ '
    s += f'& ${mean_bd_rates[midx, 5:9].mean():.3f}\%$ & ${mean_bd_psnrs[midx, 5:9].mean():.3f}$ '
    midx += 1
    print(s + '\\\\')  

    print('\t\\midrule')
    s = 'BitMore (D16) '
    s += f'& ${mean_bd_rates[midx, :5].mean():.3f}\%$ & ${mean_bd_psnrs[midx, :5].mean():.3f}$ '
    s += f'& ${mean_bd_rates[midx, 9:13].mean():.3f}\%$ & ${mean_bd_psnrs[midx, 9:13].mean():.3f}$ '
    s += f'& ${mean_bd_rates[midx, 5:9].mean():.3f}\%$ & ${mean_bd_psnrs[midx, 5:9].mean():.3f}$ '
    midx += 1

    print(s + '\\\\')
    s = 'ABCD (EDSR) '
    s += f'& ${mean_bd_rates[midx, :5].mean():.3f}\%$ & ${mean_bd_psnrs[midx, :5].mean():.3f}$ '
    s += f'& ${mean_bd_rates[midx, 9:13].mean():.3f}\%$ & ${mean_bd_psnrs[midx, 9:13].mean():.3f}$ '
    s += f'& ${mean_bd_rates[midx, 5:9].mean():.3f}\%$ & ${mean_bd_psnrs[midx, 5:9].mean():.3f}$ '
    midx += 1

    print(s + '\\\\')
    s = 'ABCD (Swin) '
    s += f'& ${mean_bd_rates[midx, :5].mean():.3f}\%$ & ${mean_bd_psnrs[midx, :5].mean():.3f}$ '
    s += f'& ${mean_bd_rates[midx, 9:13].mean():.3f}\%$ & ${mean_bd_psnrs[midx, 9:13].mean():.3f}$ '
    s += f'& ${mean_bd_rates[midx, 5:9].mean():.3f}\%$ & ${mean_bd_psnrs[midx, 5:9].mean():.3f}$ '
    midx += 1
    print(s + '\\\\')
    print('\t\\bottomrule')
    print('\\end{tabular}')

    # Only BD-Rate
    midx = 0
    print('\\begin{tabular}{lrrr}')
    print('\t\\toprule')
    print('\tAgainst Method & GF-2 & GF-6 PMS & GF-6 WFI \\\\')
    print('\t\\midrule')
    s = 'Baseline '
    s += f'& ${mean_bd_rates[midx, :5].mean():.3f}\%$ '
    s += f'& ${mean_bd_rates[midx, 9:13].mean():.3f}\%$ '
    s += f'& ${mean_bd_rates[midx, 5:9].mean():.3f}\%$ '
    midx += 1
    print(s + '\\\\')
    print('\t\\midrule')

    s = 'JPEG 2000 '
    s += f'& ${mean_bd_rates[midx, :5].mean():.3f}\%$ '
    s += f'& ${mean_bd_rates[midx, 9:13].mean():.3f}\%$ '
    s += f'& ${mean_bd_rates[midx, 5:9].mean():.3f}\%$ '
    midx += 1
    print(s + '\\\\')
    
    s = 'JPEG XL '
    s += f'& ${mean_bd_rates[midx, :5].mean():.3f}\%$ '
    s += f'& ${mean_bd_rates[midx, 9:13].mean():.3f}\%$ '
    s += f'& ${mean_bd_rates[midx, 5:9].mean():.3f}\%$ '
    midx += 1
    print(s + '\\\\')  

    s = 'DLPR\_nll '
    s += f'& ${mean_bd_rates[midx, :5].mean():.3f}\%$ '
    s += f'& ${mean_bd_rates[midx, 9:13].mean():.3f}\%$ '
    s += f'& ${mean_bd_rates[midx, 5:9].mean():.3f}\%$ '
    midx += 1
    print(s + '\\\\')  

    print('\t\\midrule')
    s = 'BitMore (D16) '
    s += f'& ${mean_bd_rates[midx, :5].mean():.3f}\%$ '
    s += f'& ${mean_bd_rates[midx, 9:13].mean():.3f}\%$ '
    s += f'& ${mean_bd_rates[midx, 5:9].mean():.3f}\%$ '
    midx += 1

    print(s + '\\\\')
    s = 'ABCD (EDSR) '
    s += f'& ${mean_bd_rates[midx, :5].mean():.3f}\%$ '
    s += f'& ${mean_bd_rates[midx, 9:13].mean():.3f}\%$ '
    s += f'& ${mean_bd_rates[midx, 5:9].mean():.3f}\%$ '
    midx += 1

    print(s + '\\\\')
    s = 'ABCD (Swin) '
    s += f'& ${mean_bd_rates[midx, :5].mean():.3f}\%$ '
    s += f'& ${mean_bd_rates[midx, 9:13].mean():.3f}\%$ '
    s += f'& ${mean_bd_rates[midx, 5:9].mean():.3f}\%$ '
    midx += 1
    print(s + '\\\\')
    print('\t\\bottomrule')
    print('\\end{tabular}')

    for I in range(13):
        plt.figure(figsize=(8, 6)) 
        plt.plot(base_bpsp_values[I], base_psnrs[I], 
                 label='Baseline', color='grey', linestyle='--', marker='x', alpha=0.6)
        plt.plot(jpeg2000_bpsp_values[I], jpeg2000_psnrs[I], 
                 label='JPEG 2000', linestyle='-', marker='o', alpha=0.9)
        plt.plot(jpegxl_bpsp_values[I], jpegxl_psnrs[I], 
                 label='JPEG XL', linestyle='-', marker='o', alpha=0.9)
        plt.plot(dlpr_bpsp_values[I], dlpr_psnrs[I], 
                 label='DLPR_nll', linestyle='-', marker='o', alpha=0.9)
        plt.plot(base_bpsp_values[I], bdr_psnrs[0][I], 
                 label='BitMore (D16)', linestyle='-', marker='o', alpha=0.9)
        plt.plot(base_bpsp_values[I], bdr_psnrs[1][I], 
                 label='ABCD (EDSR)', linestyle='-', marker='o', alpha=0.9)
        plt.plot(base_bpsp_values[I], bdr_psnrs[2][I], 
                 label='ABCD (Swin)', linestyle='-', marker='o', alpha=0.9)
        plt.plot(LBDRN_bpsp_values[I], LBDRN_psnrs[I], 
                 label='LBDRN-MSIC', linestyle='-', marker='o', alpha=0.9)
        plt.legend(loc='best')
        plt.xlabel('Bits per sub-pixel (bpsp)', fontsize=16)
        plt.ylabel('PSNR (dB)', fontsize=16)
        plt.xticks(fontsize=16) 
        plt.yticks(fontsize=16) 
        plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.4)  
        plt.savefig(f'RD_Curve_SOTA_lbr_{I}.png', dpi=300) 
        plt.close()

    plt.figure(figsize=(8, 6)) 
    plt.plot(base_bpsp_values[:5].mean(axis=0), base_psnrs[:5].mean(axis=0), 
                label='Baseline', color='grey', linestyle='--', marker='x', alpha=0.6)
    plt.plot(jpeg2000_bpsp_values[:5].mean(axis=0), jpeg2000_psnrs[:5].mean(axis=0), 
                label='JPEG 2000', linestyle='-', marker='o', alpha=0.9)
    plt.plot(jpegxl_bpsp_values[:5].mean(axis=0), jpegxl_psnrs[:5].mean(axis=0), 
                label='JPEG XL', linestyle='-', marker='o', alpha=0.9)
    plt.plot(dlpr_bpsp_values[:5].mean(axis=0), dlpr_psnrs[:5].mean(axis=0), 
                 label='DLPR_nll', linestyle='-', marker='o', alpha=0.9)
    plt.plot(base_bpsp_values[:5].mean(axis=0), bdr_psnrs[0][:5].mean(axis=0), 
                label='BitMore (D16)', linestyle='-', marker='o', alpha=0.9)
    plt.plot(base_bpsp_values[:5].mean(axis=0), bdr_psnrs[1][:5].mean(axis=0), 
                label='ABCD (EDSR)', linestyle='-', marker='o', alpha=0.9)
    plt.plot(base_bpsp_values[:5].mean(axis=0), bdr_psnrs[2][:5].mean(axis=0), 
                label='ABCD (Swin)', linestyle='-', marker='o', alpha=0.9)
    plt.plot(LBDRN_bpsp_values[:5].mean(axis=0), LBDRN_psnrs[:5].mean(axis=0), 
                label='LBDRN-MSIC', linestyle='-', marker='o', alpha=0.9)
    plt.legend(loc='best')
    plt.xlabel('Bits per sub-pixel (bpsp)', fontsize=16)
    plt.ylabel('PSNR (dB)', fontsize=16)
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16) 
    plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.4)  
    plt.savefig(f'Average_RD_Curve_SOTA_lbr.png', dpi=300) 
    plt.close()

    plt.figure(figsize=(8, 6)) 
    plt.plot(base_bpsp_values[5:9].mean(axis=0), base_psnrs[5:9].mean(axis=0), 
                label='Baseline', color='grey', linestyle='--', marker='x', alpha=0.6)
    plt.plot(jpeg2000_bpsp_values[5:9].mean(axis=0), jpeg2000_psnrs[5:9].mean(axis=0), 
                label='JPEG 2000', linestyle='-', marker='o', alpha=0.9)
    plt.plot(jpegxl_bpsp_values[5:9].mean(axis=0), jpegxl_psnrs[5:9].mean(axis=0), 
                label='JPEG XL', linestyle='-', marker='o', alpha=0.9)
    plt.plot(dlpr_bpsp_values[5:9].mean(axis=0), dlpr_psnrs[5:9].mean(axis=0), 
                 label='DLPR_nll', linestyle='-', marker='o', alpha=0.9)
    plt.plot(base_bpsp_values[5:9].mean(axis=0), bdr_psnrs[0][5:9].mean(axis=0), 
                label='BitMore (D16)', linestyle='-', marker='o', alpha=0.9)
    plt.plot(base_bpsp_values[5:9].mean(axis=0), bdr_psnrs[1][5:9].mean(axis=0), 
                label='ABCD (EDSR)', linestyle='-', marker='o', alpha=0.9)
    plt.plot(base_bpsp_values[5:9].mean(axis=0), bdr_psnrs[2][5:9].mean(axis=0), 
                label='ABCD (Swin)', linestyle='-', marker='o', alpha=0.9)
    plt.plot(LBDRN_bpsp_values[5:9].mean(axis=0), LBDRN_psnrs[5:9].mean(axis=0), 
                label='LBDRN-MSIC', linestyle='-', marker='o', alpha=0.9)
    plt.legend(loc='best')
    plt.xlabel('Bits per sub-pixel (bpsp)', fontsize=16)
    plt.ylabel('PSNR (dB)', fontsize=16)
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16) 
    plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.4)  
    plt.savefig(f'Average_RD_Curve_GF6_WFI_SOTA_lbr.png', dpi=300) 
    plt.close()

    plt.figure(figsize=(8, 6)) 
    plt.plot(base_bpsp_values[9:13].mean(axis=0), base_psnrs[9:13].mean(axis=0), 
                label='Baseline', color='grey', linestyle='--', marker='x', alpha=0.6)
    plt.plot(jpeg2000_bpsp_values[9:13].mean(axis=0), jpeg2000_psnrs[9:13].mean(axis=0), 
                label='JPEG 2000', linestyle='-', marker='o', alpha=0.9)
    plt.plot(jpegxl_bpsp_values[9:13].mean(axis=0), jpegxl_psnrs[9:13].mean(axis=0), 
                label='JPEG XL', linestyle='-', marker='o', alpha=0.9)
    plt.plot(dlpr_bpsp_values[9:13].mean(axis=0), dlpr_psnrs[9:13].mean(axis=0), 
                 label='DLPR_nll', linestyle='-', marker='o', alpha=0.9)
    plt.plot(base_bpsp_values[9:13].mean(axis=0), bdr_psnrs[0][9:13].mean(axis=0), 
                label='BitMore (D16)', linestyle='-', marker='o', alpha=0.9)
    plt.plot(base_bpsp_values[9:13].mean(axis=0), bdr_psnrs[1][9:13].mean(axis=0), 
                label='ABCD (EDSR)', linestyle='-', marker='o', alpha=0.9)
    plt.plot(base_bpsp_values[9:13].mean(axis=0), bdr_psnrs[2][9:13].mean(axis=0), 
                label='ABCD (Swin)', linestyle='-', marker='o', alpha=0.9)
    plt.plot(LBDRN_bpsp_values[9:13].mean(axis=0), LBDRN_psnrs[9:13].mean(axis=0), 
                label='LBDRN-MSIC', linestyle='-', marker='o', alpha=0.9)
    plt.legend(loc='best')
    plt.xlabel('Bits per sub-pixel (bpsp)', fontsize=16)
    plt.ylabel('PSNR (dB)', fontsize=16)
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16) 
    plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.4)  
    plt.savefig(f'Average_RD_Curve_GF6_PMS_SOTA_lbr.png', dpi=300) 
    plt.close()


def split_ratio():
    # Bit-depth reduction
    csv_file_path = f'SOTA_results/Baseline_11rps.csv'
    base_psnrs, base_bits_values, base_bpsp_values = read_csv(csv_file_path)

    # split ratio
    srs = [1, 2, 3]
    psnrs = np.zeros((len(srs), 5, 6))
    bits_values = np.zeros((len(srs), 5, 6))
    bpsp_values = np.zeros((len(srs), 5, 6))
    for i, sr in enumerate(srs):
        csv_file_path = f'outputs-rel-colors-D2/results_r{sr}_bc64_nl2_D2_prec16_lr0.001_bs8192_e10.csv'
        psnrs[i], bits_values[i], bpsp_values[i] = read_csv(csv_file_path)
    bd_psnrs = np.zeros(3)
    bd_rates = np.zeros(3)
    for i, sr in enumerate(srs):
        bd_psnrs[i] = BD_PSNR(base_bits_values.mean(axis=0), base_psnrs.mean(axis=0), 
                          bits_values[i].mean(axis=0), psnrs[i].mean(axis=0))
        print(f'BD-PSNR when sr={sr}: {bd_psnrs[i]}')
        bd_rates[i] = BD_RATE(base_bits_values.mean(axis=0), base_psnrs.mean(axis=0), 
                          bits_values[i].mean(axis=0), psnrs[i].mean(axis=0))
        print(f'BD-Rate when sr={sr}: {bd_rates[i]}')


if __name__ == "__main__":
    feature_set()
    network_hyperparameter()
    training_hyperparameter()
    SOTA()
    SOTA_lowbitrates()
    split_ratio()
