# Pyfbow , A python wrapper of [fbow](https://github.com/rmsalinas/fbow) with pybind11 

> FBOW (Fast Bag of Words) is an extremmely optimized version of the DBow2/DBow3 libraries. The library is highly optimized to speed up the Bag of Words creation using AVX,SSE and MMX instructions. In loading a vocabulary, fbow is ~80x faster than DBOW2 (see tests directory and try). In transforming an image into a bag of words using on machines with AVX instructions, it is ~6.4x faster.


Unlike the [work](https://github.com/vik748/pyfbow) by @vik748, this project only depends `opencv` (the version of 3.* and 4.* are both compatible).Instead of `boost_python`, I use [pybind11](https://github.com/pybind/pybind11.git), a header-only python wrap library to wrap python and c++ . For seek of the ablity of convertation between `cv::Mat` and `np.ndarray`, @edmBernard 's work is incorporated.  

# Build and Install

1. Clone this repo 
```shell
git clone --recurse-submodules  https://github.com/Sologala/pyfbow.git
cd pyfbow
```
2. build fbow
```shell
./build_fbow.sh
```

3. install with python
```shell
python3 setup.py develop
```
You can also install `pyfbow` into python's site-packages by
```shell
python3 setup.py install
```

# How to uninstall

After installed by python , A manifest file named `files.txt` will be created.
```shell
xargs rm -rf < files.txt
```

# Quick Start 


You can create a vocabulary by python script.
```shell
python3 tests/create_voc.py ./data -o out.voc
```