#!/usr/bin/env bash

rm nohup.out
nohup python main.py --gpu 1 --eval_step 1 --name first &