#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdint.h>
#include <nccl.h>


#define SHM_NAME      "/nccl_inspector"
#define SHM_CAPACITY  1024
#define COMM_MAGIC    0xACC11235UL

typedef enum {
    NCCL_EV_ALLREDUCE     = 0,
    NCCL_EV_ALLGATHER     = 1,
    NCCL_EV_REDUCESCATTER = 2,
    NCCL_EV_BROADCAST     = 3,
    NCCL_EV_REDUCE        = 4,
    NCCL_EV_SEND          = 5,
    NCCL_EV_RECV          = 6,
} NcclEventType;

typedef struct {
    uint64_t    timestamp_ns;   // ktime when call entered shim
    uint64_t    duration_ns;    // filled in after real NCCL returns
    uint8_t     event_type;     // NcclEventType
    uint32_t    rank;           // this process's rank
    uint32_t    nranks;         // total ranks in communicator
    uint64_t    count;          // number of elements
    uint8_t     datatype;       // ncclDataType_t
    uint8_t     op;             // ncclRedOp_t (for allreduce/reduce)
    uint8_t     algo;           // algorithm NCCL chose (read from comm internals)
    uint8_t     protocol;       // LL / LL128 / Simple
    char        comm_id[16];    // hex of communicator pointer
    int32_t     peer;           // peer rank for send/recv, root for reduce
    uint32_t    pid;            // getpid() — used to correlate RDMA events
} NcclEvent;

typedef struct {
    uint64_t    magic;
    uint32_t    write_idx;
    uint32_t    read_idx;
    uint32_t    capacity;
    uint32_t    pad;
    NcclEvent   events[SHM_CAPACITY];
} NcclShm;


static NcclShm *g_shm    = NULL;
static int      g_shm_fd = -1;

static ncclResult_t (*real_ncclAllReduce)(const void*, void*, size_t,
    ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclAllGather)(const void*, void*, size_t,
    ncclDataType_t, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclReduceScatter)(const void*, void*, size_t,
    ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclBroadcast)(const void*, void*, size_t,
    ncclDataType_t, int, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclReduce)(const void*, void*, size_t,
    ncclDataType_t, ncclRedOp_t, int, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclSend)(const void*, size_t,
    ncclDataType_t, int, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclRecv)(void*, size_t,
    ncclDataType_t, int, ncclComm_t, cudaStream_t) = NULL;


static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static void shm_init(void) {
    g_shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (g_shm_fd < 0) { perror("shm_open"); return; }

    size_t sz = sizeof(NcclShm);
    ftruncate(g_shm_fd, sz);

    g_shm = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_SHARED, g_shm_fd, 0);
    if (g_shm == MAP_FAILED) { perror("mmap"); g_shm = NULL; return; }

    if (g_shm->magic != COMM_MAGIC) {
        memset(g_shm, 0, sz);
        g_shm->magic    = COMM_MAGIC;
        g_shm->capacity = SHM_CAPACITY;
    }
    fprintf(stderr, "[nccl_shim] shared memory initialized: %s\n", SHM_NAME);
}

static void write_event(NcclEvent *ev) {
    if (!g_shm) return;
    ev->pid = (uint32_t)getpid();
    uint32_t idx = __atomic_fetch_add(&g_shm->write_idx, 1, __ATOMIC_SEQ_CST)
                   % SHM_CAPACITY;
    memcpy(&g_shm->events[idx], ev, sizeof(*ev));
}

static void resolve_syms(void) {
    real_ncclAllReduce     = dlsym(RTLD_NEXT, "ncclAllReduce");
    real_ncclAllGather     = dlsym(RTLD_NEXT, "ncclAllGather");
    real_ncclReduceScatter = dlsym(RTLD_NEXT, "ncclReduceScatter");
    real_ncclBroadcast     = dlsym(RTLD_NEXT, "ncclBroadcast");
    real_ncclReduce        = dlsym(RTLD_NEXT, "ncclReduce");
    real_ncclSend          = dlsym(RTLD_NEXT, "ncclSend");
    real_ncclRecv          = dlsym(RTLD_NEXT, "ncclRecv");
}


__attribute__((constructor))
static void shim_init(void) {
    resolve_syms();
    shm_init();
}

__attribute__((destructor))
static void shim_fini(void) {
    if (g_shm)  munmap(g_shm, sizeof(NcclShm));
    if (g_shm_fd >= 0) close(g_shm_fd);
}


/* ── Collective wrappers ─────────────────────────────────────── */

ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff,
    size_t count, ncclDataType_t datatype, ncclRedOp_t op,
    ncclComm_t comm, cudaStream_t stream)
{
    NcclEvent ev = {0};
    ev.timestamp_ns = now_ns();
    ev.event_type   = NCCL_EV_ALLREDUCE;
    ev.count        = count;
    ev.datatype     = (uint8_t)datatype;
    ev.op           = (uint8_t)op;
    ev.peer         = -1;
    ncclCommCount(comm, (int*)&ev.nranks);
    ncclCommUserRank(comm, (int*)&ev.rank);
    snprintf(ev.comm_id, sizeof(ev.comm_id), "%p", (void*)comm);

    ncclResult_t ret = real_ncclAllReduce(sendbuff, recvbuff, count,
                                          datatype, op, comm, stream);

    ev.duration_ns = now_ns() - ev.timestamp_ns;
    write_event(&ev);

    fprintf(stderr, "[nccl_shim] allreduce rank=%u/%u count=%zu dt=%u op=%u dur=%.3fus\n",
            ev.rank, ev.nranks, count, datatype, op,
            ev.duration_ns / 1000.0);
    return ret;
}

ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff,
    size_t sendcount, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream)
{
    NcclEvent ev = {0};
    ev.timestamp_ns = now_ns();
    ev.event_type   = NCCL_EV_ALLGATHER;
    ev.count        = sendcount;
    ev.datatype     = (uint8_t)datatype;
    ev.peer         = -1;
    ncclCommCount(comm, (int*)&ev.nranks);
    ncclCommUserRank(comm, (int*)&ev.rank);
    snprintf(ev.comm_id, sizeof(ev.comm_id), "%p", (void*)comm);

    ncclResult_t ret = real_ncclAllGather(sendbuff, recvbuff, sendcount,
                                           datatype, comm, stream);

    ev.duration_ns = now_ns() - ev.timestamp_ns;
    write_event(&ev);

    fprintf(stderr, "[nccl_shim] allgather rank=%u/%u count=%zu dt=%u dur=%.3fus\n",
            ev.rank, ev.nranks, sendcount, datatype,
            ev.duration_ns / 1000.0);
    return ret;
}

ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op,
    ncclComm_t comm, cudaStream_t stream)
{
    NcclEvent ev = {0};
    ev.timestamp_ns = now_ns();
    ev.event_type   = NCCL_EV_REDUCESCATTER;
    ev.count        = recvcount;
    ev.datatype     = (uint8_t)datatype;
    ev.op           = (uint8_t)op;
    ev.peer         = -1;
    ncclCommCount(comm, (int*)&ev.nranks);
    ncclCommUserRank(comm, (int*)&ev.rank);
    snprintf(ev.comm_id, sizeof(ev.comm_id), "%p", (void*)comm);

    ncclResult_t ret = real_ncclReduceScatter(sendbuff, recvbuff, recvcount,
                                               datatype, op, comm, stream);

    ev.duration_ns = now_ns() - ev.timestamp_ns;
    write_event(&ev);

    fprintf(stderr, "[nccl_shim] reducescatter rank=%u/%u count=%zu dur=%.3fus\n",
            ev.rank, ev.nranks, recvcount, ev.duration_ns / 1000.0);
    return ret;
}

ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff,
    size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream)
{
    NcclEvent ev = {0};
    ev.timestamp_ns = now_ns();
    ev.event_type   = NCCL_EV_BROADCAST;
    ev.count        = count;
    ev.datatype     = (uint8_t)datatype;
    ev.peer         = root;
    ncclCommCount(comm, (int*)&ev.nranks);
    ncclCommUserRank(comm, (int*)&ev.rank);
    snprintf(ev.comm_id, sizeof(ev.comm_id), "%p", (void*)comm);

    ncclResult_t ret = real_ncclBroadcast(sendbuff, recvbuff, count,
                                           datatype, root, comm, stream);

    ev.duration_ns = now_ns() - ev.timestamp_ns;
    write_event(&ev);

    fprintf(stderr, "[nccl_shim] broadcast rank=%u/%u count=%zu root=%d dur=%.3fus\n",
            ev.rank, ev.nranks, count, root, ev.duration_ns / 1000.0);
    return ret;
}

ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff,
    size_t count, ncclDataType_t datatype, ncclRedOp_t op,
    int root, ncclComm_t comm, cudaStream_t stream)
{
    NcclEvent ev = {0};
    ev.timestamp_ns = now_ns();
    ev.event_type   = NCCL_EV_REDUCE;
    ev.count        = count;
    ev.datatype     = (uint8_t)datatype;
    ev.op           = (uint8_t)op;
    ev.peer         = root;
    ncclCommCount(comm, (int*)&ev.nranks);
    ncclCommUserRank(comm, (int*)&ev.rank);
    snprintf(ev.comm_id, sizeof(ev.comm_id), "%p", (void*)comm);

    ncclResult_t ret = real_ncclReduce(sendbuff, recvbuff, count,
                                        datatype, op, root, comm, stream);

    ev.duration_ns = now_ns() - ev.timestamp_ns;
    write_event(&ev);

    fprintf(stderr, "[nccl_shim] reduce rank=%u/%u count=%zu root=%d dur=%.3fus\n",
            ev.rank, ev.nranks, count, root, ev.duration_ns / 1000.0);
    return ret;
}

/* ── Point-to-point wrappers ─────────────────────────────────── */

ncclResult_t ncclSend(const void *sendbuff, size_t count,
    ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream)
{
    NcclEvent ev = {0};
    ev.timestamp_ns = now_ns();
    ev.event_type   = NCCL_EV_SEND;
    ev.count        = count;
    ev.datatype     = (uint8_t)datatype;
    ev.peer         = peer;
    ncclCommCount(comm, (int*)&ev.nranks);
    ncclCommUserRank(comm, (int*)&ev.rank);
    snprintf(ev.comm_id, sizeof(ev.comm_id), "%p", (void*)comm);

    ncclResult_t ret = real_ncclSend(sendbuff, count, datatype,
                                      peer, comm, stream);

    ev.duration_ns = now_ns() - ev.timestamp_ns;
    write_event(&ev);

    fprintf(stderr, "[nccl_shim] send rank=%u/%u count=%zu peer=%d dur=%.3fus\n",
            ev.rank, ev.nranks, count, peer, ev.duration_ns / 1000.0);
    return ret;
}

ncclResult_t ncclRecv(void *recvbuff, size_t count,
    ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream)
{
    NcclEvent ev = {0};
    ev.timestamp_ns = now_ns();
    ev.event_type   = NCCL_EV_RECV;
    ev.count        = count;
    ev.datatype     = (uint8_t)datatype;
    ev.peer         = peer;
    ncclCommCount(comm, (int*)&ev.nranks);
    ncclCommUserRank(comm, (int*)&ev.rank);
    snprintf(ev.comm_id, sizeof(ev.comm_id), "%p", (void*)comm);

    ncclResult_t ret = real_ncclRecv(recvbuff, count, datatype,
                                      peer, comm, stream);

    ev.duration_ns = now_ns() - ev.timestamp_ns;
    write_event(&ev);

    fprintf(stderr, "[nccl_shim] recv rank=%u/%u count=%zu peer=%d dur=%.3fus\n",
            ev.rank, ev.nranks, count, peer, ev.duration_ns / 1000.0);
    return ret;
}
