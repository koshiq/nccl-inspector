#[path = "bpf/rdma_probe.skel.rs"]
mod rdma_probe;
use rdma_probe::*;

use anyhow::Result;
use libbpf_rs::skel::{OpenSkel, Skel, SkelBuilder};
use libbpf_rs::RingBufferBuilder;
use std::mem::MaybeUninit;
use std::time::Duration;

#[repr(C)]
struct RdmaEvent {
    timestamp_ns: u64,
    qp_num:       u32,
    bytes:        u32,
    opcode:       u32,
    pid:          u32,
    comm:         [u8; 16],
}

fn handle_event(data: &[u8]) -> i32 {
    if data.len() < std::mem::size_of::<RdmaEvent>() {
        return 1;
    }
    let e = unsafe { &*(data.as_ptr() as *const RdmaEvent) };
    let comm = std::str::from_utf8(&e.comm)
        .unwrap_or("?")
        .trim_end_matches('\0');

    println!(
        "[{:>15}ns] QP={:<6} bytes={:<8} opcode={} pid={} comm={}",
        e.timestamp_ns, e.qp_num, e.bytes, e.opcode, e.pid, comm
    );
    0
}

fn main() -> Result<()> {
    env_logger::init();

    let mut obj = MaybeUninit::uninit();
    let skel_builder = RdmaProbeSkelBuilder::default();
    let open_skel = skel_builder.open(&mut obj)?;
    let mut skel = open_skel.load()?;
    skel.attach()?;

    println!("Tracing RDMA ib_post_send... Ctrl+C to stop");

    let mut builder = RingBufferBuilder::new();
    builder.add(&skel.maps.events, handle_event)?;
    let ringbuf = builder.build()?;

    loop {
        ringbuf.poll(Duration::from_millis(100))?;
    }
}
