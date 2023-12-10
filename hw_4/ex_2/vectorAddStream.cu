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

void vectorAdditionCudaStreams(DataType *h_in1, DataType *h_in2, DataType *h_out, int len, int segmentSize, int inputLength) {
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

    //@@ Initialize the 1D grid and block dimensions here
    int dimx = 32;
    dim3 block(dimx, 1);
    dim3 grid((inputLength + block.x - 1) / block.x, 1);

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
    int segmentSizes[] = {64, 128, 256, 512};

    for (int i = 0; i < 4; ++i) {
        cudaEventRecord(start);

        vectorAdditionCudaStreams(hostInput1, hostInput2, hostOutput, inputLength, segmentSizes[i], inputLength);

        DataType elapsed_time = elapsed();

        printf("Segment Size: %d, Time: %f sec\n", segmentSizes[i], elapsed_time);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    printf("\n---------------------------------------------\n");
    printf("SUCCESS\n");

    return 0;
}

