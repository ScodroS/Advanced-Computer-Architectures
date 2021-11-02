# Advanced-Computer-Architectures
Course material for Advanced Computer Architectures course.

## Getting started with the repo

### Fork the repository

Fork the repo using the proper button, as shown below:

![Alt Text](https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/fork.png)

### Clone your fork
To be able to work on the project, you will need to clone it to the device.

Copy the path, as shown below:

![Cloning](https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/clone.png)

Create a folder with your name and surname, then 

```bash
mkdir ACA/<name>_<surname>
git clone <link copied from GitHub>
```

### Do your excercise and save the changes
Go ahead and complete the excercises. When you’re ready to submit your changes, stage and commit your changes.

In the project root folder, type:
```bash
git add .
git commit -m "<comment to the changes>"
git push
```

In this way, your excercise implementations are saved in your personal GitHub profile.

## Usage
In the excercise directory, create the build folder, launch the cmake command and compile the sources:
```bash
# Only the first time
mkdir build
cd build
cmake ..

# Every time you have to recompile
make
./<executable_name>
```

## Benchmarking

Add the following line to the main function in the source file:
```c++
std::cout << TM_host.duration() << ";" << TM_device.duration() << ";" << TM_host.duration() / TM_device.duration() << std::endl;
```

Then launch the benchmarking script in the build folder:
```bash
../../../utils/benchmark.sh <executable_name>
```

If you are using a Jetson board, make sure to set the power profile to max, typing:
```bash
sudo nvpmodel -m<n> # (The number may change, depending on the device, for TX2 <n>=0)
sudo jetson_clocks
```

## Hardware details
To have a complete overview of the Jetson hardware status, type:
```bash
sudo jtop
```

If it isn't already installed, type:
```bash
sudo -H pip3 install -U jetson-stats
sudo systemctl restart jetson_stats.service
```