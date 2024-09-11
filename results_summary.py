import os
import re
import csv
import argparse


def extract_metrics(file_path):
    performance_metrics = {}
    mse_pattern = re.compile(r'MSE: (\d+\.\d+)')
    psnr_pattern = re.compile(r'PSNR: (\d+\.\d+)')
    bpsp_pattern = re.compile(r'bpsp=(\d+\.\d+)')
    bits_pattern = re.compile(r'Total size: (\d+) bytes')
    time_pattern = re.compile(r'Time elapsed: (\d+\.\d+)')
    with open(file_path, 'r') as file:
        for line in file:
            mse_match = mse_pattern.search(line)
            psnr_match = psnr_pattern.search(line)
            bpsp_match = bpsp_pattern.search(line)
            bits_match = bits_pattern.search(line)
            time_match = time_pattern.search(line)
            if mse_match:
                performance_metrics['MSE'] = float(mse_match.group(1))
            if psnr_match:
                performance_metrics['PSNR'] = float(psnr_match.group(1))
            if bpsp_match:
                performance_metrics['bpsp'] = float(bpsp_match.group(1))
            if bits_match:
                performance_metrics['bits'] = 8 * float(bits_match.group(1))
            if time_match:
                print(f'decTime: {time_match.group(1)}')

    return performance_metrics


def extract_metrics1(file_path):
    nn_pattern = re.compile(r'nn: (\d+) bytes')
    MSB_pattern = re.compile(r'MSB: (\d+) bytes')
    time_pattern = re.compile(r'Time elapsed: (\d+\.\d+)')
    nn_bits, MSB_bits = 0, 0
    with open(file_path, 'r') as file:
        for line in file:
            nn_match = nn_pattern.search(line)
            MSB_match = MSB_pattern.search(line)
            time_match = time_pattern.search(line)
            if nn_match:
                nn_bits = 8*float(nn_match.group(1))
                print(f'nn bits: {nn_bits}')
            if MSB_match:
                MSB_bits = 8*float(MSB_match.group(1))
                print(f'MSB bits: {MSB_bits}')
            if time_match:
                print(f'encTime: {time_match.group(1)}')
        print(100*MSB_bits/(MSB_bits+nn_bits))


def parse_args():
    parser = argparse.ArgumentParser(description="Results Summary")
    parser.add_argument('-o', '--output_dir', default='outputs-rel-colors-D2', type=str,
                        help='output dir')
    parser.add_argument('-sr', '--split_ratio', type=int, default=1,
                        help='tile size (default: 1)')
    parser.add_argument('-bc', '--base_channel', type=int, default=64,
                        help='base channel (default: 64)')
    parser.add_argument('-nl', '--num_layers', type=int, default=2,
                        help='Number of layers (default: 2)')
    parser.add_argument('-D', '--D', type=int, default=2,
                        help='#neighbors (2D+1)^2')
    parser.add_argument('-prec', '--precision', type=int, default=16,
                        help=' (default: 16)')
    parser.add_argument('-lr', '--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('-bs', '--batch_size', type=int, default=8192,
                        help='batch size (default: 8192)')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    return parser.parse_args()


def save_to_csv():
    args = parse_args()

    fs = '{}/results_r{}_bc{}_nl{}_D{}_prec{}_lr{}_bs{}_e{}.csv'
    csv_file = fs.format(args.output_dir, args.split_ratio,
                                args.base_channel, args.num_layers, args.D, args.precision,
                                args.lr, args.batch_size, args.epochs)
    
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        files = [
            "TRIPLESAT_2_MS_L1_20191107021947_001FFCVI_002_0120200811001001_001",
            "TRIPLESAT_2_MS_L1_20191107021950_001FFCVI_003_0120200811001001_002",
            "TRIPLESAT_2_MS_L1_20191107021954_001FFCVI_004_0120200811001001_001",
            "TRIPLESAT_2_MS_L1_20200109023258_002115VI_003_0120200811001001_001",
            "TRIPLESAT_2_MS_L1_20200109023301_002115VI_004_0120200811001001_001"
        ]
        if csv_file == f'{args.output_dir}/results_r1_bc64_nl2_D2_prec16_lr0.001_bs8192_e10.csv':
            files = [
                "TRIPLESAT_2_MS_L1_20191107021947_001FFCVI_002_0120200811001001_001",
                "TRIPLESAT_2_MS_L1_20191107021950_001FFCVI_003_0120200811001001_002",
                "TRIPLESAT_2_MS_L1_20191107021954_001FFCVI_004_0120200811001001_001",
                "TRIPLESAT_2_MS_L1_20200109023258_002115VI_003_0120200811001001_001",
                "TRIPLESAT_2_MS_L1_20200109023301_002115VI_004_0120200811001001_001",
                "GF6_WFI_Sample_A",
                "GF6_WFI_Sample_B",
                "GF6_WFI_Sample_C",
                "GF6_WFI_Sample_D",
                "GF6_PMS_Sample_A",
                "GF6_PMS_Sample_B",
                "GF6_PMS_Sample_C",
                "GF6_PMS_Sample_D",
            ]
        metrics = ['MSE', 'PSNR', 'bpsp', 'bits']
        csv_headers = ['K'] + [f"{file}_{metric}" for file in files for metric in metrics]
        writer.writerow(csv_headers)
        # for K in range(1, 7):
        for K in range(1, 12):
            row = [f"K{K}"] + [None] * (len(files) * 4)
            for i, filename in enumerate(files):
                print(f"Processing {filename} for K={K}")
                base_str = f"{args.output_dir}/{filename}_r{args.split_ratio}_K{K}_bc{args.base_channel}_nl{args.num_layers}_D{args.D}_prec{args.precision}_lr{args.lr}_bs{args.batch_size}_e{args.epochs}"
                metrics_file = os.path.join(base_str, 'decode.txt')
                try:
                    metrics = extract_metrics(metrics_file)
                    row[4*i+1] = metrics['MSE']
                    row[4*i+2] = metrics['PSNR']
                    row[4*i+3] = metrics['bpsp']
                    row[4*i+4] = metrics['bits']
                except:
                    print('decode.txt does not exists.')
                
                if i == 0 and K == 3:
                    metrics_file = os.path.join(base_str, 'encode.txt')
                    extract_metrics1(metrics_file)
                    
            writer.writerow(row)
                
    print(f"All results have been written to {csv_file}")


if __name__ == "__main__":
    save_to_csv()