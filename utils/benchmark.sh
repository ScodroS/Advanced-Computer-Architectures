#!/bin/bash

echo "Executable path: " $1

sum_cpu=0
sum_gpu=0
sum_spe=0
total=0;

for i in {1..5}
do
  row=$(./$1 | tail -n 1)
  cpu=$(echo $row | cut -d ';' -f1 | cut -d '.' -f1  )
  gpu=$(echo $row | cut -d ';' -f2 | cut -d '.' -f1  )
  spe=$(echo $row | cut -d ';' -f3 | cut -d '.' -f1  )
  ((sum_cpu += $cpu))
  ((sum_gpu += $gpu))
  ((sum_spe += $spe))
  ((total++))
done

avg_cpu=`expr $sum_cpu / $total`
avg_gpu=`expr $sum_gpu / $total`
avg_spe=`expr $sum_spe / $total`

echo "CPU time: "$avg_cpu" ms"
echo "GPU time: "$avg_gpu" ms"
echo "speedup : "$avg_spe"x"