#include <stdio.h>
#include <cuda.h>

int *a, *b;  // host data
int *c, *c2;  // result

//Cuda error checking - non mandatory
void cudaCheckError() {
 cudaError_t e=cudaGetLastError();
 if(e!=cudaSuccess) {
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
   exit(0); 
 }
}

//GPU kernel 
__global__
void convGPU(int *A, int *B, int *C, int N){
    //Get current column and row
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int element = 0;
    //Verify if it's in the bounds
    if (i < (N+N-1)){
        //Compute an element
        for(int k = 0; k < N; k++){
            element += A[k] * (i - k < N && i - k >= 0 ? B[i - k] : 0);
        }
    }
    //Store result of the element
    C[i] = element;
}

//CPU function
void convCPU(int *A, int *B, int *C, int N){
    for (int i = 0; i < N+N - 1; ++i)
    {
        int element = 0;
        for (int k = 0; k < N; ++k)
        {
            element += A[k] * (i - k < N && i - k >= 0 ? B[i - k] : 0);
        }
        C[i] = element;
    }
}

int main(int argc,char **argv)
{
    printf("Begin \n");
    //Size of signals
    int n=150000;
    //iterations
    int m = n+n - 1;
    //Number of blocks
    int nBytes = n*sizeof(int);
    int cBytes = m*sizeof(int);
    //Block size and number
    int block_size, block_no;

    //memory allocation 
    a = (int *) malloc(nBytes);
    b = (int *) malloc(nBytes);
    c = (int *) malloc(cBytes);
    c2 = (int *) malloc(cBytes);

    int *a_d,*b_d,*c_d;
    block_size = 350; //threads per block 
    block_no = (m+1)/block_size;
    
    //Work definition
    dim3 dimBlock(block_size, 1, 1);
    dim3 dimGrid(block_no, 1, 1);

    // Data filling
    for(int i=0;i<n;i++)
        a[i]=i,b[i]=i;

    printf("\n\nAllocating device memory on host..\n");
   //GPU memory allocation
    cudaMalloc((void **) &a_d, n*sizeof(int));
    cudaMalloc((void **) &b_d, n*sizeof(int));
    cudaMalloc((void **) &c_d, m*sizeof(int));

    printf("Copying to device..\n");
    cudaMemcpy(a_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, n*sizeof(int), cudaMemcpyHostToDevice);

    //Starting clock
    clock_t start_d=clock();
    printf("Doing GPU convolution\n\n");
    convGPU<<<block_no,block_size>>>(a_d, b_d, c_d, n);
    cudaCheckError();

    //Wait for kernel call to finish
    cudaThreadSynchronize();

    clock_t end_d = clock();
    
    printf("Doing CPU convolution\n");
    clock_t start_h = clock();
    convCPU(a, b, c2, n);
    clock_t end_h = clock();
    
    //Time computing
    double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;
    double time_h = (double)(end_h-start_h)/CLOCKS_PER_SEC;

    //Copying data back to host, this is a blocking call and will not start until all kernels are finished
    cudaMemcpy(c, c_d, m*sizeof(int), cudaMemcpyDeviceToHost);
    printf("m = %d \t GPU time = %fs \t CPU time = %fs\n", m, time_d, time_h);

    //Free GPU memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    return 0;
}
