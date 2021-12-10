## 00. Fibonacci

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l6_00.jpg" width="500" height=auto> 

### Performance reference table (N = 44):
# The parallel version is worth it but only for values above ~30. Below this value we don't have speedup (we slow down the computation)

CPU time   | CPU* time  | Speedup  | Device             | Mode                   |Author
-----------| ---------- | -------- | ------------------ | ---------------------- |------
15567 ms   | 7753 ms    | 2x       | Nvidia Jetson TX2  | OpenMP (-O0)           | -
4741  ms   | 2976 ms    | 1.6x     | ryzen 3700x        | OpenMP (-O0), tasks    | ScodroS
4741  ms   | 2973 ms    | 1.6x     | ryzen 3700x        | OpenMP (-O0), sections | ScodroS

## 01. QuickSort
# 

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l6_01.jpg" width="500" height=auto> 

### Performance reference table (N = 1 << 20):

CPU time   | CPU* time | Speedup  | Device             | Mode         |Author
-----------| --------- | -------- | ------------------ | ------------ |------
23693 ms   | 6499 ms   | 3x       | Nvidia Jetson TX2  | OpenMP (-O0) | -

## 02. QuickSort

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l6_02.jpg" width="500" height=auto> 

