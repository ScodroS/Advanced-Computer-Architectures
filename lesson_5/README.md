## 01. MatrixMultiplication (OpenMP)

### Performance reference table (ROWS = COLS = 1024):

CPU time   | Parallel time | Speedup  | Device             | Mode             |Author
-----------| ------------- | -------- | ------------------ | ---------------- |------
57194 ms   | 25 ms         | 2198x    | Nvidia Jetson TX2  | Shared Mem (GPU) | -
52100 ms   | 84 ms         | 614x     | Nvidia Jetson TX2  | No Shared (GPU)  | -
30300 ms   | 8174 ms       | 3x       | Nvidia Jetson TX2  | OpenMP (-O3)     | -
35865 ms   | 9793 ms       | 3x       | Nvidia Jetson TX2  | OpenMP (-O0)     | -
1385 ms    | 535           | ~3x      | Ryzen 3700x        | OpenMP (-O3)     | ScodroS
2905 ms    | 540 ms        | ~6x      | Ryzen 3700x        | OpenMP (-O0)     | ScodroS

## 02. Factorial

### Performance reference table (N = 268435456):
# anomaly detected in the results of my solution

CPU time   | Parallel time | Speedup  | Device             | Mode  | Author
-----------| ------------- | -------- | ------------------ | ----- | ------
1845 ms    | 454 ms        | 3x       | Nvidia Jetson TX2  | OpenMP| -
680 ms     | 46 ms         | 15x      | Ryzen 3700x        | OpenMP| ScodroS

## 03. Find

Find two (given) consecutive numbers in an array.

### Performance reference table (N = 67108864):
## I'm considering the case where consecutive numbers are NOT found (which is the case where OMP parallelization works best!) so index = -1.
## Index is not always the same in sequential and parallel executions (depends on which thread finds the 2 consecutive elements first).

CPU time   | Parallel time | Speedup  | Device             | Mode  | Author
-----------| ------------- | -------- | ------------------ | ----  | ------
178 ms     | 61 ms         | 2x       | Nvidia Jetson TX2  | OpenMP| -
122 ms     | 8 ms          | ~15x     | Ryzen 3700x        | OpenMP| ScodroS

## 04. RC4 Chiper

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l5_04.jpg" width="500" height=auto> 

### Performance reference table (N = 256):
## The biggest part of the computation consists in swapping elements of the array S. This is a critical part of the process
## so I have included it inside a omp critical region. Unfortunately this badly affects the performance, that is decreasing
## with bigger amounts of threads (due to the overhead when accessing the critical region of code of the cracking part).

CPU time   | Parallel time | Speedup  | Device             | Mode               |Author
-----------| ------------- | -------- | ------------------ | ------------------ |------
72146 ms   | 63231 ms      | 1.13x    | Nvidia Jetson TX2  | OpenMP             | -
19818 ms   | 20390 ms      | 0.97x    | Ryzen 3700x        | OpenMP (2 threads) | ScodroS
19727 ms   | 21767 ms      | 0.91x    | Ryzen 3700x        | OpenMP (4 threads) | ScodroS
19685 ms   | 24837 ms      | 0.79x    | Ryzen 3700x        | OpenMP (8 threads) | ScodroS
19706 ms   | 28667 ms      | 0.69x    | Ryzen 3700x        | OpenMP (16 threads)| ScodroS
