#!/usr/bin/env bash

if [ $# -lt 3 ]
then
    echo "Usage: bash $0 CONFIG WORK_DIR GPUS"
    exit
fi

CONFIG=$1
WORK_DIR=$2
GPUS=$3

BASE_PORT=29500
INCREMENT=1

port=$BASE_PORT
isfree=$(netstat -taln | grep $port)

while [[ -n "$isfree" ]]; do
    port=$[port+INCREMENT]
    isfree=$(netstat -taln | grep $port)
done

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

if [ ${GPUS} == 1 ]; then
    python -u $(dirname "$0")/train.py  $CONFIG --work-dir=${WORK_DIR} ${@:4}
else
    python -u -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$port \
        $(dirname "$0")/train.py $CONFIG --work-dir=${WORK_DIR} --launcher pytorch ${@:4}
fi
