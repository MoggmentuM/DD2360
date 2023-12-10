#include <stdio.h>
#include <sys/time.h>

#include <time.h>
#include <stdlib.h>

#include <curand_kernel.h>
#include <curand.h>

#define DataType double
#define max_number 100
#define minimum_number 0

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < len) out[id] = in1[id] + in2[id];
}

//@@ Insert code to implement timer start
cudaEvent_t start, stop;

//@@ Insert code to implement timer stop
DataType elapsed() {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    return milliseconds / 1000.0; // convert to seconds
}

DataType cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((DataType)tp.tv_sec + (DataType)tp.tv_usec * 1.e-6);
}

void vectorAdditionCudaStreams(DataType *h_in1, DataType *h_in2, DataType *h_out, int len, int segmentSize, int inputLength, dim3 grid, dim3 block) {
    int segmentBytes = segmentSize * sizeof(DataType);

    DataType *d_in1[4], *d_in2[4], *d_out[4];

    for (int i = 0; i < 4; ++i) {
        cudaMalloc((void**)&d_in1[i], segmentBytes);
        cudaMalloc((void**)&d_in2[i], segmentBytes);
        cudaMalloc((void**)&d_out[i], segmentBytes);
    }

    cudaStream_t streams[4];
    for (int i = 0; i < 4; ++i) {
        cudaStreamCreate(&streams[i]);
        cudaMemcpyAsync(d_in1[i], h_in1 + i * segmentSize, segmentBytes, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_in2[i], h_in2 + i * segmentSize, segmentBytes, cudaMemcpyHostToDevice, streams[i]);
    }


    for (int i = 0; i < 4; ++i) {
        vecAdd<<<grid, block, 0, streams[i]>>>(d_in1[i], d_in2[i], d_out[i], segmentSize);
    }

    for (int i = 0; i < 4; ++i) {
        cudaMemcpyAsync(h_out + i * segmentSize, d_out[i], segmentBytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < 4; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    for (int i = 0; i < 4; ++i) {
        cudaFree(d_in1[i]);
        cudaFree(d_in2[i]);
        cudaFree(d_out[i]);
    }
}

int main(int argc, char **argv) {
    srand(time(NULL));

    int inputLength;
    DataType *hostInput1;
    DataType *hostInput2;
    DataType *hostOutput;

    if (argc < 2) {
        printf("You must provide at least one argument\n");
        exit(0);
    } else {
        inputLength = atoi(argv[1]);
    }

    printf("The input length is %d\n", inputLength);

    size_t size = inputLength * sizeof(DataType);

    hostInput1 = (DataType *)malloc(size);
    hostInput2 = (DataType *)malloc(size);
    hostOutput = (DataType *)malloc(size);

    for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = rand() % (max_number + 1 - minimum_number) + minimum_number;
        hostInput2[i] = rand() % (max_number + 1 - minimum_number) + minimum_number;
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Define different segment sizes for testing
    int segmentSizes[] = {min(64,inputLength/32), min(128,inputLength/16), min(256,inputLength/8), min(512,inputLength/4)};


    // optimize grid and block dimention
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
        
    int warpSize = prop.warpSize;
    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;

    int desiredBlockSize = 256;
    int optimalBlockSize = min(desiredBlockSize, maxThreadsPerSM / warpSize) * warpSize;

    int blocksPerGrid = (inputLength + optimalBlockSize - 1) / optimalBlockSize;

    dim3 block(optimalBlockSize);
    dim3 grid(blocksPerGrid);

    for (int i = 0; i < 4; ++i) {
        cudaEventRecord(start);

        vectorAdditionCudaStreams(hostInput1, hostInput2, hostOutput, inputLength, segmentSizes[i], inputLength, grid, block);

        DataType elapsed_time = elapsed();

        printf("Segment Size: %d, Time: %f sec\n", segmentSizes[i], elapsed_time);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    printf("\n---------------------------------------------\n");

    return 0;
}

