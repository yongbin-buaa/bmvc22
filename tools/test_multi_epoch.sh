#!/bin/bash

for ((i=19;i<=25;i++));
do
    checkpoint=epoch_$i.pth
    echo 'Testing '$checkpoint'...'
    CUDA_VISIBLE_DEVICES="0,1,2,3" \
    ./tools/dist_test.sh configs/textrecog/gcan/gcan_radical.py \
    work_dir/gcan_radical_v0/$checkpoint 4 --eval acc
done