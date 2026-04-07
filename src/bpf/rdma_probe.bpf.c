#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

struct rdma_event {
    __u64 timestamp_ns;
    __u64 latency_ns;
    __u32 qp_num;
    __u32 bytes;
    __u32 opcode;
    __u32 pid;
    __u8  event_type;
    char  comm[16];
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 4096 * 1024);
} events SEC(".maps");

// Hash map: qp_num -> send timestamp
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u32);
    __type(value, __u64);
} inflight SEC(".maps");

// Probe 1: fires on every RDMA packet transmission
SEC("kprobe/rxe_xmit_packet")
int BPF_KPROBE(probe_rxe_xmit_packet,
               struct ib_qp *qp,
               void *pkt,
               struct sk_buff *skb)
{
    __u32 qp_num = BPF_CORE_READ(qp, qp_num);
    __u64 ts     = bpf_ktime_get_ns();

    // Store send timestamp for latency calculation
    bpf_map_update_elem(&inflight, &qp_num, &ts, BPF_ANY);

    struct rdma_event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e)
        return 0;

    e->timestamp_ns = ts;
    e->latency_ns   = 0;
    e->qp_num       = qp_num;
    e->bytes        = BPF_CORE_READ(skb, len);
    e->opcode       = 0;
    e->pid          = bpf_get_current_pid_tgid() >> 32;
    e->event_type   = 0;
    bpf_get_current_comm(e->comm, sizeof(e->comm));

    bpf_ringbuf_submit(e, 0);
    return 0;
}

// Probe 2: fires when a competition is posted to the CQ
SEC("kprobe/rxe_cq_post")
int BPF_KPROBE(probe_rxe_cq_post,
               void *cq,
               void *cqe,
               int solicited)
{
    __u64 now = bpf_ktime_get_ns();

    __u32 qp_num   = 0;
    __u32 byte_len = 0;
    __u32 opcode   = 0;

    bpf_probe_read_kernel(&qp_num,   sizeof(qp_num),   (void *)cqe + 28);
    bpf_probe_read_kernel(&byte_len, sizeof(byte_len), (void *)cqe + 20);
    bpf_probe_read_kernel(&opcode,   sizeof(opcode),   (void *)cqe + 12);

    if (qp_num == 0)
        return 0;

    __u64 *send_ts = bpf_map_lookup_elem(&inflight, &qp_num);
    __u64 latency  = send_ts ? (now - *send_ts) : 0;

    if (send_ts)
        bpf_map_delete_elem(&inflight, &qp_num);

    struct rdma_event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e)
        return 0;

    e->timestamp_ns = now;
    e->latency_ns   = latency;
    e->qp_num       = qp_num;
    e->bytes        = byte_len;
    e->opcode       = opcode;
    e->pid          = bpf_get_current_pid_tgid() >> 32;
    e->event_type   = 1;
    bpf_get_current_comm(e->comm, sizeof(e->comm));

    bpf_ringbuf_submit(e, 0);
    return 0;
}
char LICENSE[] SEC("license") = "GPL";
