#!/usr/bin/env bash

SCRIPT=$1
GPUS=$2
PORT=${PORT:-29500}

MKL_SERVICE_FORCE_INTEL=1 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT $SCRIPT ${@:3}
# Any arguments from the third one are captured by ${@:3}
