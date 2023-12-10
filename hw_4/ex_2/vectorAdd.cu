#include <stdio.h>
#include <sys/time.h>

// libraries to generate random numbers in C
#include <time.h>
#include <stdlib.h>

#include <cstdlib> //to convert char* to int

// libraries to generate random numbers in CUDA
#include <curand_kernel.h>
#include <curand.h>

#define DataType double
#define max_number 100
#define minimum_number 0

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < len) out[id] = in1[id] + in2[id];
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

int main(int argc, char **argv) {

  srand(time(NULL));   // Initialization for random numbers, should only be called once.

  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  //DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  // decode arguments
  if(argc < 2) {
      printf("You must provide at least one argument\n");
      exit(0);
  } else {
    inputLength = atoi(argv[1]);
  }

  printf("The input length is %d\n", inputLength);

  size_t size = inputLength * sizeof(DataType);

  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType *) malloc(size);
  hostInput2 = (DataType *) malloc(size);
  hostOutput = (DataType *) malloc(size);

  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  for (int i = 0; i < inputLength; i++) {
    // generate a pseudo-random integer between minimum_number and max_number
    hostInput1[i] = rand() % (max_number + 1 - minimum_number) + minimum_number;
    hostInput2[i] = rand() % (max_number + 1 - minimum_number) + minimum_number;
  }

  //start timer
  cudaEventRecord(start);

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc((void **)&deviceInput1, size);
  cudaMalloc((void **)&deviceInput2, size);
  cudaMalloc((void **)&deviceOutput, size);

  //@@ Insert code to below to Copy memory to the GPU here
  //cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind )
  cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);

  //@@ Initialize the 1D grid and block dimensions here
  int dimx = 32;
  dim3 block(dimx, 1);
  dim3 grid((inputLength + block.x - 1) / block.x, 1);

  //@@ Launch the GPU Kernel here
  DataType start = cpuSecond();

  vecAdd<<<grid, block>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();

  DataType elapsed_time = elapsed();

  printf("vecAdd<<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, elapsed);

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  double tolerance = 1.0e-14;
  DataType expeted;
  for(int i=0; i<inputLength; i++)
  {
    expeted = hostInput1[i] + hostInput2[i];
    //if the absolute value is greater than the tolerance we have an error
    if(fabs(hostOutput[i] - expeted) > tolerance)
    {
        printf("\nError: value of hostOutput[%d] = %f instead of %f\n\n", i, hostOutput[i], expeted);
        exit(1);
    } else {
      //printf("\nOk: value of hostOutput[%d] = %f - expected: %f\n\n", i, hostOutput[i], expeted);
    }
  }

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);


  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  printf("\n---------------------------------------------\n");
  printf("SUCCESS\n");

  return 0;
}
