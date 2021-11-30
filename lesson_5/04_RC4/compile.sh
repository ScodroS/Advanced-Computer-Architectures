#!/bin/bash
DIR=`dirname $0`

g++ -std=c++14 -O3 -fopenmp ${DIR}/RC4.cpp -I"$DIR"/include/ -o rc4
./rc4
