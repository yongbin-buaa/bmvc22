#!/usr/bin/env bash

if [ $# -lt 3 ]
then
    echo "Usage: bash $0 CONFIG CHECKPOINT GPUS"
    exit
fi

BASE_PORT=29500
INCREMENT=1

port=$BASE_PORT
isfree=$(netstat -taln | grep $port)

while [[ -n "$isfree" ]]; do
    port=$[port+INCREMENT]
    isfree=$(netstat -taln | grep $port)
done

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

if [ ${GPUS} == 1 ]; then
    python -u $(dirname "$0")/test.py  $CONFIG $CHECKPOINT ${@:4}
else
    python -u -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$port \
        $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
fi
