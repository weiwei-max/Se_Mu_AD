#!/bin/bash

export CARLA_ROOT=/home/wei/dw/carla_garage/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/miniconda3/lib

export OMP_NUM_THREADS=20  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
# torchrun --nnodes=1 --nproc_per_node=1 --max_restarts=1 --rdzv_id=42353467 --rdzv_backend=c10d train.py --id 01_0410 --batch_size 16 --setting all --root_dir /media/wei/Data/dataset/dataset --logdir /home/wei/carla_garage/01_0410 --use_controller_input_prediction 1 --use_wp_gru 0 --use_discrete_command 1 --use_tp 1 --cpu_cores 8 --num_repetitions 1 --continue_epoch 0 --load_file /home/wei/carla_garage/result/001_so_128/model_0030.pth
torchrun --nnodes=1 --nproc_per_node=1 --max_restarts=1 --rdzv_id=42353467 --rdzv_backend=c10d train.py --id 01_0410 --batch_size 4 --setting all --root_dir /media/wei/Data/dataset/dataset --logdir /home/wei/carla_garage/01_0410 --use_controller_input_prediction 1 --use_wp_gru 0 --use_discrete_command 1 --use_tp 1 --cpu_cores 8 --num_repetitions 1 --continue_epoch 0 --load_file /home/wei/carla_garage/result/001_so_128/model_0030.pth
