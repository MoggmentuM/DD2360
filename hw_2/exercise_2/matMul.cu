#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>


#define DataType float

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){

	 //@@ Insert code to implement matrix multiplication here
    int rows = blockIdx.y * blockDim.y + threadIdx.y;
    int cols = blockIdx.x * blockDim.x + threadIdx.x;


    if(rows < numARows && cols < numBColumns){
        DataType sum = 0;
        for(int i = 0; i < numAColumns; i++){
            sum += A[rows * numAColumns + i] * B[i * numBColumns + cols];
        }
        C[rows * numBColumns + cols] = sum;
    }
}


int main(int argc, char **argv) {

    DataType *hostA; // The A matrix
    DataType *hostB; // The B matrix
    DataType *hostC; // The output C matrix
    DataType *resultRef; // The reference result
    DataType *deviceA;
    DataType *deviceB;
    DataType *deviceC;
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;
    int numCColumns;

    //@@ Insert code below to read in numARows, numAColumns, numBColumns and numBRows from args
    if(argc != 4){
        printf("Usage: %s numARows numAColumns numBColumns\n", argv[0]);
        exit(1);
    }
    numARows = atoi(argv[1]);
    numAColumns = atoi(argv[2]);
    numBColumns = atoi(argv[3]);
    numBRows = numAColumns;
    numCRows = numARows;
    numCColumns = numBColumns;
    if(numARows <= 0 || numAColumns <= 0 || numBColumns <= 0){
        printf("Invalid arguments! \n");
        exit(1);
    }

     printf("Input matrix dimensions: A(%d x %d), B(%d x %d), C(%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);


    //@@ Insert code below to allocate Host memory for input and output
    hostA = (DataType*)malloc(numARows * numAColumns * sizeof(DataType));
    hostB = (DataType*)malloc(numBRows * numBColumns * sizeof(DataType));

    //Checking in order to interrupt memory allocation didnt work.
    if(hostA == NULL || hostB == NULL){
        printf("Allocating memory failed\n");
        exit(1);
    }


    //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU



    for(int i = 0; i < numARows; i++) {
        for(int j = 0; j < numAColumns; j++) {
            hostA[i * numAColumns + j] = (DataType)rand() / RAND_MAX;
        }
    }

    for(int i = 0; i < numBRows; i++) {
        for(int j = 0; j < numBColumns; j++) {
            hostB[i * numBColumns + j] = (DataType)rand() / RAND_MAX;
        }
    }

    // Allocate memory for resultRef
    resultRef = (DataType*)malloc(numCRows * numCColumns * sizeof(DataType));
    if (resultRef == NULL) {
        printf("Error allocating memory for resultRef\n");
        exit(1);
    }

    // Compute the reference result in CPU


    for(int i = 0; i < numARows; i++) {
        for(int j = 0; j < numBColumns; j++) {
            resultRef[i * numBColumns + j] = 0; // Initialize the element to 0
            for(int k = 0; k < numAColumns; k++) {
                // Accumulate the sum for the dot product of row i from A and column j from B
                resultRef[i * numBColumns + j] += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
            }
        }
    }



    hostC = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));
    if(hostA == NULL || hostB == NULL || hostC == NULL || resultRef == NULL){
        printf("Error allocating memory\n");
        exit(1);
    }

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(DataType));
    cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(DataType));
    cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(DataType));

    //@@ Insert code to below to Copy memory to the GPU here

    timeval timer;

    gettimeofday(&timer, NULL);
    double start_memtogpu = timer.tv_sec * 1000000 + timer.tv_usec;


    cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);

    gettimeofday(&timer, NULL);
    double end_memtogpu = timer.tv_sec * 1000000 + timer.tv_usec;

    double elapsed_time_memtogpu = (end_memtogpu - start_memtogpu) / 1e6;
    printf("Copy memory to GPU: %f s\n", elapsed_time_memtogpu);

    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid((numBColumns - 1) / 32 + 1, (numARows - 1) / 32 + 1, 1);
    dim3 dimBlock(32, 32, 1);

    //@@ Launch the GPU Kernel here

    gettimeofday(&timer, NULL);
    double start_gpukernel = timer.tv_sec * 1000000 + timer.tv_usec;
    gemm<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
    cudaDeviceSynchronize();

    gettimeofday(&timer, NULL);
    double end_gpukernel = timer.tv_sec * 1000000 + timer.tv_usec;

    double elapsed_time_gpukernel = (end_gpukernel - start_gpukernel) / 1e6;
    printf("Time to execute the GPU Kernel: %f s\n", elapsed_time_gpukernel);


    //@@ Copy the GPU memory back to the CPU here

    gettimeofday(&timer, NULL);
    double start_gputocpu = timer.tv_sec * 1000000 + timer.tv_usec;
    cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);

    gettimeofday(&timer, NULL);
   double end_gputocpu = timer.tv_sec * 1000000 + timer.tv_usec;
    double elapsed_time_gputocpu = (end_gputocpu - start_gputocpu) / 1e6;
    printf("Time to copy from GPU memory to CPU: %f s\n", elapsed_time_gputocpu);




    //@@ Insert code below to compare the output with the reference
    bool error = false;
    for(int i = 0; i < numCRows; i++) {
        for(int j = 0; j < numCColumns; j++) {
            if(abs(hostC[i * numCColumns + j] - resultRef[i * numCColumns + j]) > 1e-2){
                error = true;
                break;
            }
        }
    }

    if(error) {
        printf("The results are incorrect!\n");
    } else {
        printf("The results are correct.\n");
    }



    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);



    //@@ Free the CPU memory here
    free(hostA);
    free(hostB);
    free(hostC);
    free(resultRef);



    return 0;
}

