// SPDX-License-Identifier: GPL-2.0
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

struct rdma_event {
    __u64 timestamp_ns;
    __u32 qp_num;
    __u32 bytes;
    __u32 opcode;
    __u32 pid;
    char  comm[16];
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 4096 * 1024);
} events SEC(".maps");

// rxe_xmit_packet(struct rxe_qp *qp, struct rxe_pkt_info *pkt, struct sk_buff *skb)
// rxe_qp embeds ib_qp as first member so pointer is compatible
// skb->len gives us actual bytes on the wire
SEC("kprobe/rxe_xmit_packet")
int BPF_KPROBE(probe_rxe_xmit_packet,
               struct ib_qp *qp,
               void *pkt,
               struct sk_buff *skb)
{
    struct rdma_event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e)
        return 0;

    e->timestamp_ns = bpf_ktime_get_ns();
    e->pid          = bpf_get_current_pid_tgid() >> 32;
    e->qp_num       = BPF_CORE_READ(qp, qp_num);
    e->bytes        = BPF_CORE_READ(skb, len);
    e->opcode       = 0;

    bpf_get_current_comm(e->comm, sizeof(e->comm));
    bpf_ringbuf_submit(e, 0);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
