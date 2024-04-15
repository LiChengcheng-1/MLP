#! /bin/bash

read -p "Please type what you want : Train(True) or Test(False)：" mode

if [ "$mode" = true ]; then
    echo "start trainning..."
    python ../src/main.py --mode $mode
elif [ "$mode" = false ]; then
    echo "start testing..."
    python ../src/main.py --mode $mode
else
    echo "Error: Invalid mode parameter"
fi