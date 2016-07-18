#ifndef CBP_COUNT_SKETCH_H_
#define CBP_COUNT_SKETCH_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
typedef TTypes<float, 2>::ConstTensor ConstFloatMatrix;
typedef TTypes<float, 2>::Tensor FloatMatrix;
typedef TTypes<int, 1>::ConstTensor IntVector;

namespace functor {

    // Perhaps... for device polymorphism?
    template<typename Device> struct CountSketch {
        void operator()(const Device& d, int batch_size, int input_size, int dims,
                        ConstFloatMatrix probs, IntVector h, IntVector s, FloatMatrix sketch);
    };

    template<typename Device> struct CountSketchGrad {
        void operator()(const Device& d, int batch_size, int input_size, int dims,
                        ConstFloatMatrix probs, IntVector h, IntVector s, FloatMatrix sketch);
    };

} // namespace functor
#endif  // CBP_COUNT_SKETCH_H_