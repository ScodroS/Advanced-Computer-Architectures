#!/bin/bash
DIR=`dirname $0`

g++ -O3 -std=c++11 -fopenmp -std=c++14 -I"$DIR"/include "$DIR"/main.cpp "$DIR"/src/MatrixMultiplication.cpp -o matrix_mul_O3

./matrix_mul_O3
#g++-8 -O0 -std=c++14 -fopenmp -I"$DIR"/include "$DIR"/main.cpp "$DIR"/src/MatrixMultiplication.cpp -o matrix_mul_O0