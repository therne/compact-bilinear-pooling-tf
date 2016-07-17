#ifndef CBP_COUNT_SKETCH_H_
#define CBP_COUNT_SKETCH_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template<typename Device> struct CountSketch {
    void operator()(const Device& d, int batch_size, int input_size, int dims, typename TTypes<float, 2>::ConstTensor probs,
                    typename TTypes<int, 1>::ConstTensor h, typename TTypes<int, 1>::ConstTensor s,
                    typename TTypes<float, 2>::Tensor sketch) {
        for (int i=0; i<batch_size; i++) {
            for (int j=0; j<dims; j++) sketch(i, j) = 0; // zero init
            for (int j=0; j<input_size; j++) {
                sketch(i, h(j)) += s(j) * probs(i, j);
            }
        }
    }
};

template<typename Device> struct CountSketchGrad {
    void operator()(const Device& d, int batch_size, int input_size, int dims, typename TTypes<float, 2>::ConstTensor probs,
                    typename TTypes<int, 1>::ConstTensor h, typename TTypes<int, 1>::ConstTensor s,
                    typename TTypes<float, 2>::Tensor sketch) {

        for (int i=0; i<batch_size; i++) {
            for (int j=0; j<dims; j++) sketch(i, j) = 0; // zero init
            for (int j=0; j<dims; j++) {
                sketch(i, j) = s(j) * probs(i, h(j));
            }
        }
    }
};

} // namespace functor
#endif  // CBP_COUNT_SKETCH_H_