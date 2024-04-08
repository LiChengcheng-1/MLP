#! /bin/bash
# shellcheck disable=SC2164
cd "$(dirname $0)" #turn the route to this file

python ../src/main.py \
--batch_size 256 \
--epoch 100 \
--experiment_number 20