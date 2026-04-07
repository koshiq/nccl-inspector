#include <stdio.h>
#include <stdlib.h>
#include <nccl.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(cmd) do { \
    cudaError_t e = cmd; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define CHECK_NCCL(cmd) do { \
    ncclResult_t r = cmd; \
    if (r != ncclSuccess) { \
        fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__, \
                ncclGetErrorString(r)); \
        exit(1); \
    } \
} while(0)

int main() {
    // Single GPU
    int dev = 0;
    CHECK_CUDA(cudaSetDevice(dev));

    ncclComm_t comm;
    ncclUniqueId id;
    CHECK_NCCL(ncclGetUniqueId(&id));
    CHECK_NCCL(ncclCommInitRank(&comm, 1, id, 0));

    // Allocate buffers
    size_t count = 1024 * 1024; // 1M floats = 4MB
    float *sendbuf, *recvbuf;
    CHECK_CUDA(cudaMalloc(&sendbuf, count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&recvbuf, count * sizeof(float)));
    CHECK_CUDA(cudaMemset(sendbuf, 1, count * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    printf("Running 5 allreduce iterations...\n");
    for (int i = 0; i < 5; i++) {
        CHECK_NCCL(ncclAllReduce(sendbuf, recvbuf, count,
                                 ncclFloat, ncclSum, comm, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        printf("  iter %d done\n", i);
    }

    printf("Running 3 allgather iterations...\n");
    for (int i = 0; i < 3; i++) {
        CHECK_NCCL(ncclAllGather(sendbuf, recvbuf, count / 1,
                                 ncclFloat, comm, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        printf("  iter %d done\n", i);
    }

    CHECK_CUDA(cudaFree(sendbuf));
    CHECK_CUDA(cudaFree(recvbuf));
    CHECK_CUDA(cudaStreamDestroy(stream));
    ncclCommDestroy(comm);

    printf("Done.\n");
    return 0;
}
