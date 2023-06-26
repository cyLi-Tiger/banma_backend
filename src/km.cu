#include <cuda_runtime_api.h>
#include <names.h>
#include <km.h>
#include <vector>

namespace triton { namespace backend { namespace NAMESPACE {

namespace {
__global__ void km_kernel(float* vec1, float* vec2, float* cost, 
                          float const* matrix, float epsilon) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        vec1[id] = matrix[id];
        vec2[id] = matrix[id];
        atomicAdd(cost, matrix[id]);
    }    
}
}

// void km(float* vec1, float* vec2, float cost, 
//         float const* matrix, float epsilon) {

// }



const int size = 32;

int main() {
    std::vector<std::vector<float>> inp(size, std::vector<float>(size, 6));
    float *h_vec1 = new float[size];
    float *h_vec2 = new float[size];
    float *h_cost = new float[1];

    float* matrix, *vec1, *vec2, *cost;
    cudaMalloc((float**)&matrix, size * size * sizeof(float));
    cudaMalloc((float**)&vec1, size * sizeof(float));
    cudaMalloc((float**)&vec2, size * sizeof(float));
    cudaMalloc((float**)&cost, 1 * sizeof(float));
    
    for (int i = 0; i < size; i++) {
        cudaMemcpy(matrix + i * size, &inp[i][0], size * sizeof(float), cudaMemcpyHostToDevice);
    }
    int block_size = 32;
    int grid_size = size * size / block_size;
    km_kernel<<<block_size, grid_size>>>(vec1, vec2, cost, matrix, 1e-4);
    cudaMemcpy(h_vec1, vec1, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vec2, vec2, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cost, cost, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(matrix);
    cudaFree(vec1);
    cudaFree(vec2);
    cudaFree(cost);
    delete[] h_vec1;
    delete[] h_vec2;
    delete[] h_cost;
}

}}}