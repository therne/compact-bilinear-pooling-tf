
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda_helper.h"
#include "count_sketch.h"

// GPU Kernel
__global__ void CountSketchKernel(int nthreads, int input_size, const float* probs,
                                  const int* h, const int* s, float* sketch) {
    CUDA_1D_KERNEL_LOOP(p, nthreads) {
        int i = p / input_size;
        int j = p % input_size;
        int projected_index = i * input_size + h[j];
        sketch[projected_index] = sketch[projected_index] + s[j] * probs[p];
    }
}

__global__ void CountSketchGradKernel(int nthreads, int dims, const float* probs,
                                      const int* h, const int* s, float* sketch) {
    CUDA_1D_KERNEL_LOOP(p, nthreads) {
        int i = p / dims;
        int j = p % dims;
        int projected_index = i * dims + h[j];
        sketch[p] = s[j] * probs[projected_index];
    }
}

__global__ void ZeroKernel(int nthreads, float* sketch) {
    CUDA_1D_KERNEL_LOOP(p, nthreads) { sketch[p] = 0; }
}

namespace functor {

template<> void CountSketch<GPUDevice>::operator()(const GPUDevice& d,
                    int batch_size,
                    int input_size,
                    int dims,
                    ConstFloatMatrix probs,
                    IntVector h,
                    IntVector s,
                    FloatMatrix sketch) {

    CudaLaunchConfig cfg = GetCudaLaunchConfig(probs.size(), d);
    ZeroKernel<<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(cfg.virtual_thread_count, sketch.data());

    CountSketchKernel<<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(cfg.virtual_thread_count,
            input_size, probs.data(), h.data(), s.data(), sketch.data());
}

template<> void CountSketchGrad<GPUDevice>::operator()(const GPUDevice& d,
                    int batch_size,
                    int input_size,
                    int dims,
                    ConstFloatMatrix probs,
                    IntVector h,
                    IntVector s,
                    FloatMatrix sketch) {

    CudaLaunchConfig cfg = GetCudaLaunchConfig(sketch.size(), d);
    ZeroKernel<<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(cfg.virtual_thread_count, sketch.data());

    CountSketchGradKernel<<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(cfg.virtual_thread_count,
            dims, probs.data(), h.data(), s.data(), sketch.data());
}


} // namespace functor
#endif // GOOGLE_CUDA