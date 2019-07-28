#!/usr/bin/env bash
for entry in `ls /home/neil/leondelee/pointmatching/PointGCN/3DSmoothNet/data/modelnet_ply_file/`
do ./3DSmoothNet -f /home/neil/leondelee/pointmatching/PointGCN/3DSmoothNet/data/modelnet_ply_file/$entry -o ./data/modelnet/  \
&& echo /home/neil/leondelee/pointmatching/PointGCN/3DSmoothNet/data/modelnet_ply_file/$entry
done