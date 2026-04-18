// Two-rank NCCL harness. Spawns two processes (fork + exec so the child
// gets a clean CUDA context — a naked fork after NCCL/CUDA init fails
// with "initialization error" in the child). Both ranks share GPU 0;
// rank 0 generates the ncclUniqueId and drops it into /tmp for the child.
//
// To route traffic through rxe (software RDMA) so the eBPF probes fire:
//   env NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_HCA=rxe0 \
//       LD_PRELOAD=$PWD/libnccl_shim.so ./test_harness_mp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include <nccl.h>
#include <cuda_runtime.h>

#define ID_PATH "/tmp/nccl_inspector.id"

#define CHECK_CUDA(cmd) do {                                             \
    cudaError_t e = cmd;                                                 \
    if (e != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                cudaGetErrorString(e));                                  \
        exit(1);                                                         \
    }                                                                    \
} while (0)

#define CHECK_NCCL(cmd) do {                                             \
    ncclResult_t r = cmd;                                                \
    if (r != ncclSuccess) {                                              \
        fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__,    \
                ncclGetErrorString(r));                                  \
        exit(1);                                                         \
    }                                                                    \
} while (0)

static void run_rank(int rank, int nranks, ncclUniqueId id) {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    // Pin each rank to its own GPU when available; fall back to device 0
    // on single-GPU boxes (NCCL will refuse to init, which is the point —
    // the harness is meant for multi-GPU nodes).
    int dev = (device_count > rank) ? rank : 0;
    CHECK_CUDA(cudaSetDevice(dev));
    fprintf(stderr, "[rank %d] using cudaDev %d of %d\n", rank, dev, device_count);

    ncclComm_t comm;
    CHECK_NCCL(ncclCommInitRank(&comm, nranks, id, rank));

    size_t count = 1024 * 1024; // 4 MiB of f32
    float *sendbuf, *recvbuf;
    CHECK_CUDA(cudaMalloc(&sendbuf, count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&recvbuf, count * sizeof(float)));
    CHECK_CUDA(cudaMemset(sendbuf, 1, count * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    fprintf(stderr, "[rank %d] 5x allreduce\n", rank);
    for (int i = 0; i < 5; i++) {
        CHECK_NCCL(ncclAllReduce(sendbuf, recvbuf, count,
                                 ncclFloat, ncclSum, comm, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    fprintf(stderr, "[rank %d] 3x allgather\n", rank);
    for (int i = 0; i < 3; i++) {
        CHECK_NCCL(ncclAllGather(sendbuf, recvbuf, count / nranks,
                                 ncclFloat, comm, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    fprintf(stderr, "[rank %d] 3x broadcast (root=0)\n", rank);
    for (int i = 0; i < 3; i++) {
        CHECK_NCCL(ncclBroadcast(sendbuf, recvbuf, count,
                                 ncclFloat, 0, comm, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    fprintf(stderr, "[rank %d] 3x reducescatter\n", rank);
    for (int i = 0; i < 3; i++) {
        CHECK_NCCL(ncclReduceScatter(sendbuf, recvbuf, count / nranks,
                                     ncclFloat, ncclSum, comm, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    CHECK_CUDA(cudaFree(sendbuf));
    CHECK_CUDA(cudaFree(recvbuf));
    CHECK_CUDA(cudaStreamDestroy(stream));
    ncclCommDestroy(comm);
    fprintf(stderr, "[rank %d] done\n", rank);
}

static int read_id_blocking(ncclUniqueId *out) {
    for (int i = 0; i < 200; i++) { // ~10s
        FILE *f = fopen(ID_PATH, "rb");
        if (f) {
            size_t n = fread(out, sizeof(*out), 1, f);
            fclose(f);
            if (n == 1) return 0;
        }
        usleep(50000);
    }
    return -1;
}

int main(int argc, char **argv) {
    int rank;
    ncclUniqueId id;
    pid_t child = -1;

    if (argc >= 2 && strcmp(argv[1], "--child") == 0) {
        rank = 1;
        if (read_id_blocking(&id) != 0) {
            fprintf(stderr, "[rank 1] timed out waiting for %s\n", ID_PATH);
            return 1;
        }
    } else {
        rank = 0;
        CHECK_NCCL(ncclGetUniqueId(&id));
        FILE *f = fopen(ID_PATH, "wb");
        if (!f) { perror("open id file"); return 1; }
        fwrite(&id, sizeof(id), 1, f);
        fclose(f);

        child = fork();
        if (child < 0) { perror("fork"); return 1; }
        if (child == 0) {
            // Fresh process image so the child gets a clean CUDA context.
            execl("/proc/self/exe", argv[0], "--child", (char *)NULL);
            perror("execl");
            _exit(1);
        }
    }

    run_rank(rank, 2, id);

    if (rank == 0 && child > 0) {
        int status = 0;
        waitpid(child, &status, 0);
        unlink(ID_PATH);
    }
    return 0;
}
