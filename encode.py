import os
import sys
import time
import random
import argparse
import subprocess
import tracemalloc

import fpzip
import torch
import numpy as np
from osgeo import gdal
from ignite.engine import Events
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import logger
from LBDRNloss import LBDRNLoss
from LBDRNmodel import LBDRNModel
from LBDRNperformance import LBDRNPerformance
from LBDRNdataset import LBDRNDataset, split_image
from modified_ignite_engine import create_supervised_evaluator, create_supervised_trainer


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
Bitstream structure:
header
for i 
    for j
        nn_i_j
        base_i_j
"""
def write_image_header(header_path, 
                       split_ratio, width, height, 
                       K, bc, nl, D, 
                       nn_bytes_list, base_bytes_list):
    n_bytes_header  = 0
    n_bytes_header += 1                           # Number of bytes header
    n_bytes_header += 1                           # split_ratio
    n_bytes_header += 2                           # width
    n_bytes_header += 2                           # height
    n_bytes_header += 1                           # K (4bits) + D (4bits)
    n_bytes_header += 1                           # log2(bc) (4bits), nl (4bits) 
    n_bytes_header += 3 * len(nn_bytes_list)      # Number of bytes nn
    n_bytes_header += 4 * len(base_bytes_list)    # Number of bytes base
    byte_to_write   = b''
    byte_to_write  += n_bytes_header.to_bytes(1, byteorder='big', signed=False)
    byte_to_write  += split_ratio.to_bytes(1, byteorder='big', signed=False)
    byte_to_write  += width.to_bytes(2, byteorder='big', signed=False)
    byte_to_write  += height.to_bytes(2, byteorder='big', signed=False)
    byte_to_write  += (K * 2 ** 4 + D).to_bytes(1, byteorder='big', signed=False)
    byte_to_write  += (int(np.log2(bc)) * 2 ** 4 + nl).to_bytes(1, byteorder='big', signed=False)
    for nn_bytes in nn_bytes_list:
        byte_to_write += nn_bytes.to_bytes(3, byteorder='big', signed=False)
    for base_bytes in base_bytes_list:
        byte_to_write += base_bytes.to_bytes(4, byteorder='big', signed=False)
    with open(header_path, 'wb') as fout: fout.write(byte_to_write)
    if n_bytes_header != os.path.getsize(header_path):
        raise ValueError(f'Invalid number of bytes in header! '
                         f'expected {n_bytes_header}, got {os.path.getsize(header_path)}')


def train(args):
    dataset = LBDRNDataset(args)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=32, pin_memory=True)  
    model = LBDRNModel(dim_in=dataset.n_feature, 
                     dim_hidden=args.base_channel, 
                     dim_out=dataset.channels,
                     num_layers=args.num_layers,
                     # activation=torch.nn.ReLU() # Default: Sine
                     )
    model = model.to(DEVICE)
    logger.log.info(model)
    for param_tensor in model.state_dict():
        logger.log.info('{}\t {}'.format(param_tensor, model.state_dict()[param_tensor].size()))
    total_params = sum(p.numel() for p in model.parameters())
    logger.log.info('total_params: {}'.format(total_params))

    optimizer = Adam(model.parameters(), lr=args.lr) 
    scheduler = lr_scheduler.StepLR(optimizer, step_size=max(1, int(args.epochs/3)), gamma=0.1)
    loss_func = LBDRNLoss()
    trainer = create_supervised_trainer(model, optimizer, loss_func, device=DEVICE)
    evaluator = create_supervised_evaluator(model, metrics={'LBDRN_performance': LBDRNPerformance()}, device=DEVICE)
    writer = SummaryWriter(log_dir=args.output_dir)
    global best_val_criterion, best_epoch
    best_val_criterion, best_epoch = 1e6, -1 # MSE
    filename = os.path.splitext(os.path.basename(args.path))[0]
    @trainer.on(Events.ITERATION_COMPLETED)
    def iter_event_function(engine):
        writer.add_scalar(f'train/loss/{filename}', engine.state.output, engine.state.iteration)
    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_event_function(engine):
        scheduler.step()
        global best_val_criterion, best_epoch
        if args.epochs == 1:
            torch.save(model.state_dict(), f'{args.output_dir}/model.pt')
            best_epoch = engine.state.epoch
            return
        if engine.state.epoch % min(args.val_duration, args.epochs) == 0: 
            evaluator.run(train_loader)
            performance = evaluator.state.metrics
            writer.add_scalar(f'val/MSE/{filename}', performance['MSE'], engine.state.epoch)
            val_criterion = performance['MSE']
            if val_criterion < best_val_criterion: 
                torch.save(model.state_dict(), f'{args.output_dir}/model.pt')
                best_val_criterion = val_criterion
                best_epoch = engine.state.epoch
                logger.log.info('Save current best val model (MSE: {:.5f}) @epoch {}'
                                .format(best_val_criterion, best_epoch))
            else:
                logger.log.info('Model is not updated (MSE: {:.5f}) @epoch: {}'
                                .format(val_criterion, engine.state.epoch))           
    @trainer.on(Events.COMPLETED)
    def final_testing_results(engine):
        logger.log.info('best epoch: {}'.format(best_epoch))
        model.load_state_dict(torch.load(f'{args.output_dir}/model.pt'))
        subprocess.call(f'rm -f {args.output_dir}/model.pt', shell=True)
        params = None
        for param_tensor in model.state_dict(): #
            if params is None:
                params = model.state_dict()[param_tensor].data.to('cpu').numpy().reshape(-1)
            else:
                params = np.concatenate((params, model.state_dict()[param_tensor].data.to('cpu').numpy().reshape(-1)))
        compressed_bytes = fpzip.compress(params, precision=args.precision, order='C')
        nn_path = f'{args.output_dir}/{filename}_nn.bin'
        with open(nn_path, 'wb') as f: f.write(compressed_bytes)
        nn_bytes = os.path.getsize(nn_path)
        nn_bpsp = nn_bytes * 8 / dataset.n_subpixels
        logger.log.info(f'nn: {nn_bytes} bytes, bpsp={nn_bpsp}')
        base_path = f'{args.output_dir}/{filename}_base.tif'
        jp2_path = f'{args.output_dir}/{filename}_base.jp2'
        cmd_encode = f"gdal_translate -of JP2OpenJPEG -co QUALITY=100 -co REVERSIBLE=YES {base_path} {jp2_path}"
        r = sh(cmd_encode) 
        logger.log.info(r)

        if False:
            org_img = gdal.Open(args.path).ReadAsArray() # CHW
            base_img = gdal.Open(f'{args.output_dir}/{filename}_base.tif').ReadAsArray() # CHW
            base_img = base_img.astype(np.uint16) << args.K
            mse_value = np.mean((org_img.astype(np.float32) - base_img.astype(np.float32)) ** 2) #
            logger.log.info(f"MSB MSE: {mse_value}")
            peak = 10000 # np.max(org_img) # 
            psnr = 10 * np.log10(peak ** 2 / mse_value)
            logger.log.info(f"MSB PSNR: {psnr}")   

        subprocess.call(f'rm -f {base_path}', shell=True)
        base_bytes = os.path.getsize(jp2_path)
        base_bpsp = base_bytes * 8 / dataset.n_subpixels
        logger.log.info(f"MSB: {base_bytes} bytes: bpsp={base_bpsp}")
        writer.close ()  

    trainer.run(train_loader, max_epochs=args.epochs)


def sh(cmd, input=''): 
    rst = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE, input=input.encode('utf-8'))
    assert rst.returncode == 0, rst.stderr.decode('utf-8')
    return rst.stdout.decode('utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LBDRN-MSIC')
    parser.add_argument('--seed', type=int, default=19920517)
    parser.add_argument('-rn', '--randomness', action='store_true',
                        help='Allow randomness during training?')
    parser.add_argument('-i', '--path', type=str,
                        help='path of input tif or img file')
    parser.add_argument('-o', '--output_dir', default='outputs', type=str,
                        help='output dir')
    parser.add_argument('-sr', '--split_ratio', type=int, default=1,
                        help='tile size (default: 1)')
    parser.add_argument('-K', '--K', type=int, default=5,
                        help=' (default: 5)')
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
    parser.add_argument('-vd', '--val_duration', type=int, default=1,
                        help='number of epoch duration for val (default: 1)')
    args = parser.parse_args()

    # tracemalloc.start()

    if not args.randomness:
        torch.manual_seed(args.seed)  #
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)
    torch.utils.backcompat.broadcast_warning.enabled = True

    org_path = args.path
    filename = os.path.splitext(os.path.basename(org_path))[0]
    fs = '{}/{}_r{}_K{}_bc{}_nl{}_D{}_prec{}_lr{}_bs{}_e{}'
    args.output_dir = fs.format(args.output_dir, filename, args.split_ratio, args.K,
                                args.base_channel, args.num_layers, args.D, args.precision,
                                args.lr, args.batch_size, args.epochs)
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    bitstream_path = f'{args.output_dir}/{filename}.bin'
    if os.path.exists(f'{args.output_dir}/encode.txt'):
        encoded = False
        with open(f'{args.output_dir}/encode.txt', 'r') as file:
            content = file.read()
            if "Time elapsed" in content:
                encoded = True
                print('Bitstream already created!')
        if encoded and os.path.exists(bitstream_path):
            sys.exit()
    logger.create_logger(args.output_dir, 'encode.txt')
    start_time = time.time()
    header_path = f'{bitstream_path}_header'
    dataset = gdal.Open(org_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    if args.split_ratio > 1:
        split_image(args.path, args.output_dir, args.split_ratio)
        for i in range(args.split_ratio):
            for j in range(args.split_ratio):
                args.path = f'{args.output_dir}/tile_{i}_{j}.tif'
                logger.log.info(args)
                train(args)
                subprocess.call(f'rm -f {args.path}', shell=True)
        nn_bytes_list, base_bytes_list = [], []
        for i in range(args.split_ratio):
            for j in range(args.split_ratio):
                sub_nn_bitstream_path = f'{args.output_dir}/tile_{i}_{j}_nn.bin'
                nn_bytes_list.append(os.path.getsize(sub_nn_bitstream_path))
                sub_base_bitstream_path = f'{args.output_dir}/tile_{i}_{j}_base.jp2'
                base_bytes_list.append(os.path.getsize(sub_base_bitstream_path))
        subprocess.call(f'rm -f {header_path}', shell=True)
        write_image_header(header_path, args.split_ratio, width, height,
                           args.K, args.base_channel, args.num_layers, args.D,
                           nn_bytes_list, base_bytes_list
                           )
        subprocess.call(f'rm -f {bitstream_path}', shell=True)
        subprocess.call(f'cat {header_path} >> {bitstream_path}', shell=True)
        subprocess.call(f'rm -f {header_path}', shell=True)

        for i in range(args.split_ratio):
            for j in range(args.split_ratio):
                sub_nn_bitstream_path = f'{args.output_dir}/tile_{i}_{j}_nn.bin'
                sub_base_bitstream_path = f'{args.output_dir}/tile_{i}_{j}_base.jp2'
                subprocess.call(f'cat {sub_nn_bitstream_path} >> {bitstream_path}', shell=True)
                subprocess.call(f'rm -f {sub_nn_bitstream_path}', shell=True)
                subprocess.call(f'cat {sub_base_bitstream_path} >> {bitstream_path}', shell=True)
                subprocess.call(f'rm -f {sub_base_bitstream_path}', shell=True)
    else:
        logger.log.info(args)
        train(args)
        nn_bitstream_path = f'{args.output_dir}/{filename}_nn.bin'
        nn_bytes_list = [os.path.getsize(nn_bitstream_path)]
        base_bitstream_path = f'{args.output_dir}/{filename}_base.jp2'
        base_bytes_list = [os.path.getsize(base_bitstream_path)]
        subprocess.call(f'rm -f {header_path}', shell=True)
        write_image_header(header_path, args.split_ratio, width, height,
                           args.K, args.base_channel, args.num_layers, args.D,
                           nn_bytes_list, base_bytes_list
                           )
        subprocess.call(f'rm -f {bitstream_path}', shell=True)
        subprocess.call(f'cat {header_path} >> {bitstream_path}', shell=True)
        subprocess.call(f'rm -f {header_path}', shell=True)
        subprocess.call(f'cat {nn_bitstream_path} >> {bitstream_path}', shell=True)
        subprocess.call(f'rm -f {nn_bitstream_path}', shell=True)
        subprocess.call(f'cat {base_bitstream_path} >> {bitstream_path}', shell=True)
        subprocess.call(f'rm -f {base_bitstream_path}', shell=True)
    
    end_time = time.time()
    logger.log.info(f'Time elapsed: {end_time - start_time}')

    # current, peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # logger.log.info(f"Current memory usage: {current / 10**6:.2f} MB")
    # logger.log.info(f"Peak memory usage: {peak / 10**6:.2f} MB")
