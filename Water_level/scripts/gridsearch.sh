#! /bin/bash
#python ../src/main.py \
#--batch_size 256 \
#--epoch 100 \
#--experiment_number 20


learning_rates=(0.001 0.01 0.1)
batch_size=(340 170 85)
hidden_size=(20 40 60 80)

for lr in "${learning_rates[@]}"; do
    for batch_size in "${batch_size[@]}"; do
        for hidden_size in "${hidden_size[@]}";do
          python main.py --lr $lr --batch_size $batch_size --hidden_size $hidden_size --project_name "MLP_Water_level_gridsearch" --mode True
      done
  done
done