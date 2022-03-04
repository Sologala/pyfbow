#!/bin/bash
CWD=`pwd`
fbow_src_dir="thirdparty/fbow"
fbow
mkdir -p $fbow_src_dir/build 
mkdir -p install_local
cd $fbow_src_dir/build


cmake -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_INSTALL_PREFIX=$CWD/install_local \
      -DCMAKE_CXX_FLAGS="-fPIC" \
      -DCMAKE_C_FLAGS="-fPIC" \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON .. && make -j$(nproc) && make install 
cd ../..
