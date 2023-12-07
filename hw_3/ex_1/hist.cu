#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
    // Shared memory for histogram bins
    extern __shared__ unsigned int s_bins[];

    // Initialize shared memory bins
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        s_bins[i] = 0;
    }

    __syncthreads();

    // Compute histogram using shared memory and atomics
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elements; i += blockDim.x * gridDim.x) {
        atomicAdd(&s_bins[input[i]], 1);
    }

    __syncthreads();

    // Write back to global memory
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&bins[i], s_bins[i]);
    }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
    // Clean up bins that saturate at 127
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    bins[idx] = (bins[idx] > 127) * 127 + (bins[idx] <= 127) * bins[idx];
}

double get_time() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main(int argc, char **argv) {

    int inputLength;
    unsigned int *hostInput;
    unsigned int *hostBins;
    unsigned int *resultRef;
    unsigned int *deviceInput;
    unsigned int *deviceBins;

    // Read in inputLength from args
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <inputLength>\n", argv[0]);
        exit(1);
    }
    inputLength = atoi(argv[1]);

    printf("The input length is %d\n", inputLength);

    // Allocate Host memory for input and output
    hostInput = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
    hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
    resultRef = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));

    // Initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, NUM_BINS - 1);

    for (int i = 0; i < inputLength; i++) {
        hostInput[i] = distribution(generator);
    }

    // Create reference result in CPU with saturation at 127
    for (int i = 0; i < NUM_BINS; i++) {
        resultRef[i] = 0;
    }

    for (int i = 0; i < inputLength; i++) {
        resultRef[hostInput[i]] = (resultRef[hostInput[i]] + 1 > 127) ? 127 : resultRef[hostInput[i]] + 1;
    }

    // Init timer
    double timer = get_time();

    // Allocate GPU memory using unified memory
    cudaMallocManaged((void **)&deviceInput, inputLength * sizeof(unsigned int));
    cudaMallocManaged((void **)&deviceBins, NUM_BINS * sizeof(unsigned int));

    // Copy memory to the GPU
    cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

    // Initialize the grid and block dimensions
    int TPB = 1024;
    dim3 blocks_in_grid_1((inputLength + TPB - 1) / TPB);
    dim3 threads_in_block_1(TPB);

    // Launch the GPU Kernel
    histogram_kernel<<<blocks_in_grid_1, threads_in_block_1, NUM_BINS * sizeof(unsigned int)>>>(
        deviceInput, deviceBins, inputLength, NUM_BINS);
    cudaDeviceSynchronize();

    // Initialize the second grid and block dimensions
    dim3 blocks_in_grid_2((NUM_BINS + TPB - 1) / TPB);
    dim3 threads_in_block_2(TPB);

    // Launch the second GPU Kernel
    convert_kernel<<<blocks_in_grid_2, threads_in_block_2>>>(deviceBins, NUM_BINS);
    cudaDeviceSynchronize();

    // Copy the GPU memory back to the CPU
    cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Output for CSV
    printf("HostBins\n");
for (int i = 0; i < NUM_BINS; i++) {
    printf("%d\n", hostBins[i]);
}

    // Get execution time
    printf("Execution time,%f\n", get_time() - timer);

    printf("End of execution\n");


    return 0;
}
