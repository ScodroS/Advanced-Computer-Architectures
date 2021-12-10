## 00. Fibonacci

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l6_00.jpg" width="500" height=auto> 

### Performance reference table (N = 44):
The parallel version is worth it but only for values above ~30. Below this value we don't have speedup (we slow down the computation)

CPU time   | CPU* time  | Speedup  | Device             | Mode                   |Author
-----------| ---------- | -------- | ------------------ | ---------------------- |------
15567 ms   | 7753 ms    | 2x       | Nvidia Jetson TX2  | OpenMP (-O0)           | -
4741  ms   | 2976 ms    | 1.6x     | ryzen 3700x        | OpenMP (-O0), tasks    | ScodroS
4741  ms   | 2973 ms    | 1.6x     | ryzen 3700x        | OpenMP (-O0), sections | ScodroS

## 01. QuickSort
 

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l6_01.jpg" width="500" height=auto> 

### Performance reference table (N = 1 << 20):

CPU time   | CPU* time | Speedup  | Device             | Mode                   |Author
-----------| --------- | -------- | ------------------ | ---------------------- |------
23693 ms   | 6499 ms   | 3x       | Nvidia Jetson TX2  | OpenMP (-O0)           | -
5768 ms    | 4612 ms   | 1.3x     | ryzen 3700x        | OpenMP (-O0), sections | -

## 02. Producer-consumer

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l6_02.jpg" width="500" height=auto> 

- Can be implemented with critical regions. In particular we have to give the same name to the regions of producer and consumer
in order to allow the execution of only one thread at a time (since the variable "count" is modified in consumer and producer).
- Can be implemented with locks. We need just one lock shared between the 2 code blocks of producer and consumer.
- Cannot be implemented with atomic (at least not without heavily modifying the code) because atomic does not support array indexing.
- Cannot be implemented with barriers only as they are used to prevent threads to continue further in the execution than the others,
not to avoid race conditions. If both threads enter the producer/consumer at the same time the behavior is undefined as they will try to
edit the array at the same time.
- Cannot be implemented with only shared variable, since we are editing "count" multiple times and we would not be able to avoid race conditions
in this case too.

 