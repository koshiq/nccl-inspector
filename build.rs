use std::path::Path;
use libbpf_cargo::SkeletonBuilder;

fn main() {
    let src = "src/bpf/rdma_probe.bpf.c";
    let out = "src/bpf/rdma_probe.skel.rs";
    SkeletonBuilder::new()
        .source(src)
        .build_and_generate(Path::new(out))
        .unwrap();
    println!("cargo:rerun-if-changed={}", src);
}
