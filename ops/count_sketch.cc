
#define EIGEN_USE_THREADS

#include <random>
#include "count_sketch.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"

using namespace tensorflow;

REGISTER_OP("CountSketch")
    .Attr("T: {float, int32}")
    .Input("probs: T")
    .Input("h: int32")
    .Input("s: int32")
    .Input("proj_size: int32")
    .Output("sketch: T");

REGISTER_OP("CountSketchGrad")
    .Attr("T: {float, int32}")
    .Input("probs: T")
    .Input("h: int32")
    .Input("s: int32")
    .Input("orig_size: int32")
    .Output("grad: T");


template <typename Device>
class CountSketchOp: public OpKernel {
public:
    explicit CountSketchOp(OpKernelConstruction* context) : OpKernel(context) {}
    
    void Compute(OpKernelContext* context) override {
        const Tensor& probs = context->input(0);
        const Tensor& h = context->input(1);
        const Tensor& s = context->input(2);
        const int32 proj_size = context->input(3).scalar<int32>()();
        
        TensorShape input_shape = probs.shape();
        auto batch_size = input_shape.dim_size(0), input_size = input_shape.dim_size(1);
        
        TensorShape output_shape;
        output_shape.AddDim(batch_size);
        output_shape.AddDim(proj_size);

        Tensor* sketch = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &sketch));

        // fill sketch
        functor::CountSketch<Device> functor;
        functor(context->eigen_device<Device>(), batch_size, input_size, proj_size, probs.matrix<float>(),
                                    h.vec<int>(), s.vec<int>(), sketch->matrix<float>());
    }
};

// CounterSketch CPU kernel
namespace functor {
template<> void CountSketch<CPUDevice>::operator()(const CPUDevice& d, int batch_size,
                int input_size, int dims, ConstFloatMatrix probs,
                IntVector h, IntVector s, FloatMatrix sketch) {

    for (int i=0; i<batch_size; i++) {
        for (int j=0; j<dims; j++) sketch(i, j) = 0; // zero init
        for (int j=0; j<input_size; j++) {
            sketch(i, h(j)) += s(j) * probs(i, j);
        }
    }
}
}

template <typename Device>
class CountSketchGradOp: public OpKernel {
public:
    explicit CountSketchGradOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& probs = context->input(0);
        const Tensor& h = context->input(1);
        const Tensor& s = context->input(2);
        const int32 proj_size = context->input(3).scalar<int32>()();

        TensorShape input_shape = probs.shape();
        auto batch_size = input_shape.dim_size(0), input_size = input_shape.dim_size(1);

        TensorShape output_shape;
        output_shape.AddDim(batch_size);
        output_shape.AddDim(proj_size);

        Tensor* sketch = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &sketch));

        // fill sketch
        functor::CountSketchGrad<Device> functor;
        functor(context->eigen_device<Device>(), batch_size, input_size, proj_size, probs.matrix<float>(),
                                    h.vec<int>(), s.vec<int>(), sketch->matrix<float>());
    }
};

// CounterSketchGrad CPU kernel
namespace functor {
template<> void CountSketchGrad<CPUDevice>::operator()(const CPUDevice& d, int batch_size,
                int input_size, int dims, ConstFloatMatrix probs,
                IntVector h, IntVector s, FloatMatrix sketch) {

    for (int i=0; i<batch_size; i++) {
        for (int j=0; j<dims; j++) sketch(i, j) = 0; // zero init
        for (int j=0; j<dims; j++) {
            sketch(i, j) = s(j) * probs(i, h(j));
        }
    }
}
}

REGISTER_KERNEL_BUILDER(Name("CountSketch").Device(DEVICE_CPU), CountSketchOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("CountSketchGrad").Device(DEVICE_CPU), CountSketchGradOp<CPUDevice>);

// TODO: FUCK CUDA!!!!
//REGISTER_KERNEL_BUILDER(Name("CountSketch").Device(DEVICE_GPU).HostMemory("proj_size"), CountSketchOp<GPUDevice>);
//REGISTER_KERNEL_BUILDER(Name("CountSketchGrad").Device(DEVICE_GPU).HostMemory("proj_size"), CountSketchGradOp<GPUDevice>);