#!/bin/bash
DIR=`dirname $0`

nvcc -w -std=c++11 -arch=sm_62 "$DIR"/MatrixMultiplication_TP.cu -I"$DIR"/include -o matrix_mul
