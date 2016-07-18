#!/usr/bin/env bash
echo Building Native ops...
TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
# TODO: GPU support
#nvcc -std=c++11 -c -o count_sketch.cu.o count_sketch.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -D_MWAITXINTRIN_H_INCLUDED -g
#g++ -std=c++11 -shared -o count_sketch.so count_sketch.cc count_sketch.cu.o -fPIC -lcudart -I $TF_INC -D_GLIBCXX_USE_CXX11_ABI=0 -g

mkdir -p build
g++ -std=c++11 -shared -o build/count_sketch.so ops/count_sketch.cc -fPIC -I $TF_INC -D_GLIBCXX_USE_CXX11_ABI=0