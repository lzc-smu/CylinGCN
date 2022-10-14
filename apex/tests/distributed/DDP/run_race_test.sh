#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=2 ddp_race_condition_test.py
