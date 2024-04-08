#! /bin/bash
# shellcheck disable=SC2164
cd "$(dirname $0)" #turn the route to this file

python ../src/main.py \
--batch_size 256 \
--epoch 100 \
<<<<<<< HEAD
--experiment_number 20
=======
>>>>>>> 2f6387746555ee267c83b136b16d37e2beb3f118
