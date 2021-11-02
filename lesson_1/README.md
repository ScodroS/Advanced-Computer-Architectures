## Test machine

CPU          | GPU       |
-------------| --------- | 
Ryzen 3700x  | RTX 2060S |

## 01. VectorAdd

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l1_01.jpg" width="500" height=auto> 

### Performance reference table (N = 1 000 000):

CPU time | GPU time | Speedup |Device             |
---------| -------- | ------- | ------------------|
894 ms   | 29 ms    | 29x     | Nvidia Jetson TX2 |
2.4 ms   | 0.1 ms   | 32x     | RTX 2060S         | 

## 02. 1DStencil

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l1_02.jpg" width="500" height=auto> 

### Performance reference table (N = 100 000 000, RADIUS=7, THREADS_PER_BLOCK=256):
CPU time   | GPU time | Speedup  | Mode |Device             |
-----------| -------- | -------- | ---- |------------------ |
13640 ms   | 92 ms    | 149x     | no_SM|Nvidia Jetson TX2  |
2820 ms    | 4 ms     | 780x     | no_SM|RTX 2060S          |
2811 ms    | 4 ms     |  744x    |  SM  |RTX 2060S          | 

### Performance reference table (N = 100 000 000, RADIUS=32, THREADS_PER_BLOCK=256):
 CPU time  | GPU time| Speedup   | Mode |Device             |
-----------| -------- | -------- | ---- |------------------ |
13159 ms   | 11 ms    | 1232x    | no_SM|RTX 2060S          |
13188 ms   | 10 ms    | 1332x    |  SM  |RTX 2060S          | 

### Performance reference table (N = 100 000 000, RADIUS=64, THREADS_PER_BLOCK=256):
 CPU time  | GPU time| Speedup   | Mode |Device             |
-----------| -------- | -------- | ---- |------------------ |
27770 ms   | 21 ms    | 1351x    | no_SM|RTX 2060S          |
27722 ms   | 18 ms    | 1549x    |  SM  |RTX 2060S          | 

## 03. MatrixMultiplication

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l1_03.jpg" width="500" height=auto> 

### Performance reference table (N = 1024, BLOCK_SIZE = 16x16):
CPU time   | GPU time | Speedup  | Mode   |   Device          |
-----------| -------- | -------- | -------| ----------------- |
52100 ms   | 84 ms    | 614x     | no_SM  |Nvidia Jetson TX2  |
2700 ms    | 5 ms     | 618x     | no_SM  |RTX 2060S          |
52100 ms   | 84 ms    | 614x     | SM     |Nvidia Jetson TX2  |
2700 ms    | 3 ms     | 992x     | SM     |RTX 2060S          | 

### Performance reference table (N = 2048, BLOCK_SIZE = 32x32):
CPU time   | GPU time | Speedup  | Mode   |   Device          |
-----------| -------- | -------- | -------| ----------------- |
466817 ms  | 217 ms   | 2155x    | SM     |Nvidia Jetson TX2  |
65770 ms   | 26ms     | 2488x    | no_SM  |RTX 2060S          |
59621 ms   | 21ms     | 2799x    | SM     |RTX 2060S          |

## 04. MatrixTranspose

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l1_04.jpg" width="500" height=auto> 

### Performance reference table (N = 1024, BLOCK_SIZE = 16x16):
CPU time   | GPU time  | Speedup  | Mode      | Device            |
-----------| --------- | -------- | --------- |------------------ |
5474 ms    | 65 ms     | 82x      | no_SM     |Nvidia Jetson TX2  |
2.8 ms     | <0.1 ms   | 64x      | no_SM     |RTX 2060S          | 

### Performance reference table (N = 8192, BLOCK_SIZE = 32x32):
CPU time   | GPU time  | Speedup  | Mode      | Device            |
-----------| --------- | -------- | --------- |------------------ |
5474 ms    | 65 ms     | 82x      | SM        |Nvidia Jetson TX2  |
743 ms     | 6 ms      | 132x     | no_SM     |RTX 2060S          |
750 ms     | 4 ms      | 183x     | SM        |RTX 2060S          | 