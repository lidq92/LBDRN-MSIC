#!/bin/bash

DEVICE_ID=$1
D=$2
BC=$3
NL=$4
LR=$5
BS=$6
EPOCH=$7
SR=$8
PREC="16"
OUTPUT_DIR=$9

file_paths=(
    "data/GF-dataset/GF-2/TRIPLESAT_2_MS_L1_20191107021947_001FFCVI_002_0120200811001001_001.tif"
    "data/GF-dataset/GF-2/TRIPLESAT_2_MS_L1_20191107021950_001FFCVI_003_0120200811001001_002.tif"
    "data/GF-dataset/GF-2/TRIPLESAT_2_MS_L1_20191107021954_001FFCVI_004_0120200811001001_001.tif"
    "data/GF-dataset/GF-2/TRIPLESAT_2_MS_L1_20200109023258_002115VI_003_0120200811001001_001.tif"
    "data/GF-dataset/GF-2/TRIPLESAT_2_MS_L1_20200109023301_002115VI_004_0120200811001001_001.tif"
    "data/GF-dataset/GF-6/GF6-WFI/GF6_WFI_Sample_A.tif"
    "data/GF-dataset/GF-6/GF6-WFI/GF6_WFI_Sample_B.tif"
    "data/GF-dataset/GF-6/GF6-WFI/GF6_WFI_Sample_C.tif"
    "data/GF-dataset/GF-6/GF6-WFI/GF6_WFI_Sample_D.tif"
    "data/GF-dataset/GF-6/GF6-PMS/GF6_PMS_Sample_A.tif"
    "data/GF-dataset/GF-6/GF6-PMS/GF6_PMS_Sample_B.tif"
    "data/GF-dataset/GF-6/GF6-PMS/GF6_PMS_Sample_C.tif"
    "data/GF-dataset/GF-6/GF6-PMS/GF6_PMS_Sample_D.tif"
)
for INPUT_PATH in "${file_paths[@]}"; do
    echo "Processing file: $INPUT_PATH"
    FILENAME=$(basename "$INPUT_PATH")
    FILENAME_NO_EXT="${FILENAME%.*}"

    for ((K=1; K<7; K++)); do
    # for ((K=1; K<12; K++)); do
        echo "Running command for K=$K"
        CUDA_VISIBLE_DEVICES=$DEVICE_ID python encode.py -K $K -i $INPUT_PATH -D $D -bc $BC -nl $NL -lr $LR -bs $BS -e $EPOCH -sr $SR -prec $PREC -o $OUTPUT_DIR
        BASE_STR="${OUTPUT_DIR}/${FILENAME_NO_EXT}_r${SR}_K${K}_bc${BC}_nl${NL}_D${D}_prec${PREC}_lr${LR}_bs${BS}_e${EPOCH}"
        CUDA_VISIBLE_DEVICES=$DEVICE_ID python decode.py -i $BASE_STR/${FILENAME_NO_EXT}.bin -org $INPUT_PATH
    done

done

echo "All files processed."