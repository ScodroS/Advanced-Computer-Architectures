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

### Performance reference table (N = 100 000 000):
CPU time   | GPU time | Speedup  | Device             |
-----------| -------- | -------- | ------------------ |
13640 ms   | 92 ms    | 149x     | Nvidia Jetson TX2  |
2820 ms    | 4 ms     | 780x     | RTX 2060S          | 

## 03. MatrixMultiplication

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l1_03.jpg" width="500" height=auto> 

### Performance reference table (N = 1024):
CPU time   | GPU time | Speedup  | Device             |
-----------| -------- | -------- | ------------------ |
52100 ms   | 84 ms    | 614x     | Nvidia Jetson TX2  |
2700 ms    | 5 ms     | 618x     | RTX 2060S          | 

## 04. MatrixTranspose

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l1_04.jpg" width="500" height=auto> 

### Performance reference table (N = 1024):
CPU time   | GPU time | Speedup  | Device             |
-----------| -------- | -------- | ------------------ |
5474 ms   | 65 ms     | 82x     | Nvidia Jetson TX2   |
2.8 ms    | <0.1 ms   | 64.2x   | RTX 2060S           | 

## 05. MatrixMultiplication (with shared memory optimization)

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l1_03.jpg" width="500" height=auto> 

### Performance reference table (N = 1024):
CPU time   | GPU time | Speedup  | Device             | Block size (=Tile width) |
-----------| -------- | -------- | ------------------ | ------------------------ |
2700 ms     | 3 ms    | 992x     | RTX 2060S          | 16                       |
2702 ms     | 3 ms    | 965x     | RTX 2060S          | 32                       |

## 06. 1DStencil (with shared memory optimization)

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l1_02.jpg" width="500" height=auto> 

### Performance reference table (N = 100 000 000):
CPU time   | GPU time | Speedup  | Device             |
-----------| -------- | -------- | ------------------ |
894 ms     | 29 ms    | 29x     | RTX 2060S           | 