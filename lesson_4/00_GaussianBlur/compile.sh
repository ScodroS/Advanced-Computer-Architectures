#!/bin/bash
DIR=`dirname $0`

# /home/accounts/studenti/id915ohp/Scrivania/opencv/include/
# nvcc -w -std=c++11 -arch=sm_62 "$DIR"/GaussianBlur.cu -I/usr/local/MATLAB/R2021b/toolbox/vision/builtins/src/ocvcg/opencv/include/ -I"$DIR"/include -o gaussian_blur             
nvcc -w -std=c++11 -arch=sm_62 "$DIR"/GaussianBlur.cu -I/home/accounts/studenti/id915ohp/Scrivania/opencv/include/ -I/ -I"$DIR"/include -o gaussian_blur
