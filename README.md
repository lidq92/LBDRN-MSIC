# LBDRN-MSIC
Lightweight Bit-Depth Recovery Network for Gaofen Satellite Multispectral Image Compression

## Requirements
See `requirements.txt`

- CUDA 12.1 + CuDNN 8.9.2
- PyTorch+Ignite
- GDAL
- fpzip

```bash
pip install -r requirements.txt
```

## Encode
```bash
CUDA_VISIBLE_DEVICES=0 python encode.py  -K 5 -i data/sample.tif -D 2 -bc 64 -nl 2 -lr 0.001 -bs 8192 -e 10 -sr 1 -prec 16 -o outputs-rel-colors-D2
```

## Decode
```bash
CUDA_VISIBLE_DEVICES=0 python decode.py -i outputs-rel-colors-D2/sample_r1_K5_bc64_nl2_D2_prec16_lr0.001_bs8192_e10/sample.bin -org data/sample.tif
```

## Experiments
```bash
# ./run.sh $DEVICE_ID $D $BC $NL $LR $BS $EPOCH $SR $OUTPUT_DIR

# USE_COORDINATES = False, USE_COLORS = True, RELATIVE = True
./run.sh 3 1 64 2 0.001 8192 10 1 outputs-rel-colors-D1
./run.sh 3 2 64 2 0.001 8192 10 1 outputs-rel-colors-D2 # K = 1 to 12
./run.sh 3 3 64 2 0.001 8192 10 1 outputs-rel-colors-D3

./run.sh 3 2 128 1 0.001 8192 10 1 outputs-rel-colors-D2
./run.sh 3 2 128 2 0.001 8192 10 1 outputs-rel-colors-D2
./run.sh 3 2 256 2 0.001 8192 10 1 outputs-rel-colors-D2

./run.sh 3 2 64 2 0.01 8192 10 1 outputs-rel-colors-D2
./run.sh 3 2 64 2 0.0001 8192 10 1 outputs-rel-colors-D2
./run.sh 3 2 64 2 0.001 4096 10 1 outputs-rel-colors-D2
./run.sh 3 2 64 2 0.001 2048 10 1 outputs-rel-colors-D2
./run.sh 3 2 64 2 0.001 8192 1 1 outputs-rel-colors-D2
./run.sh 3 2 64 2 0.001 8192 5 1 outputs-rel-colors-D2
./run.sh 3 2 64 2 0.001 8192 15 1 outputs-rel-colors-D2

./run.sh 3 2 64 2 0.001 8192 10 2 outputs-rel-colors-D2
./run.sh 3 2 64 2 0.001 8192 10 3 outputs-rel-colors-D2

# Modify constants.py
# USE_COORDINATES = True, EMBEDDING = False, USE_COLORS = True, RELATIVE = True
./run.sh 3 2 64 2 0.001 8192 10 1 outputs-coords-rel-colors-D2

# USE_COORDINATES = True, EMBEDDING = False, USE_COLORS = False
./run.sh 3 2 64 2 0.001 8192 10 1 outputs-coords

# USE_COORDINATES = True, EMBEDDING = True, USE_COLORS = False
./run.sh 3 2 64 2 0.001 8192 10 1 outputs-coords-embedding

# USE_COORDINATES = False, USE_COLORS = True, RELATIVE = False
./run.sh 3 2 64 2 0.001 8192 10 1 outputs-abs-colors-D2
./run.sh 3 0 64 2 0.001 8192 10 1 outputs-abs-colors-D0

# 
# python SOTA.py
# python SOTA_BDR.py
# ...

### Result Summary
# python results_summary.py -o outputs-rel-colors-D2 -e 10 -bc 64 -nl 2 -D 2
# ...


### SOTA compared methods (See compared_methods ...)
# python visu_image.py
# python BD_metrics.py

```

## TODO
- **Modularization**: To improve code clarity and avoid repetitive patterns, consider adding reusable functions.

## FAQ

## Contact
Dingquan Li @ Pengcheng Laboratory, dingquanli AT pku DOT edu DOT cn.