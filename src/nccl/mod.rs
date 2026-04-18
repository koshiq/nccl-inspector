// src/nccl/mod.rs
use std::fs::OpenOptions;
use std::os::unix::io::IntoRawFd;

const SHM_NAME: &str = "/dev/shm/nccl_inspector";
const SHM_CAPACITY: usize = 1024;
const COMM_MAGIC: u64 = 0xACC11235;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NcclEvent {
    pub timestamp_ns: u64,
    pub duration_ns:  u64,
    pub event_type:   u8,
    pub rank:         u32,
    pub nranks:       u32,
    pub count:        u64,
    pub datatype:     u8,
    pub op:           u8,
    pub algo:         u8,
    pub protocol:     u8,
    pub comm_id:      [u8; 16],
    pub peer:         i32,
    pub pid:          u32,
}

#[repr(C)]
struct NcclShm {
    magic:     u64,
    write_idx: u32,
    read_idx:  u32,
    capacity:  u32,
    pad:       u32,
    events:    [NcclEvent; SHM_CAPACITY],
}

pub struct NcclReader {
    shm:      *mut NcclShm,
    read_idx: u32,
}

// Safety: the shared memory pointer is only accessed through &mut self methods
unsafe impl Send for NcclReader {}

impl NcclReader {
    pub fn open() -> Option<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(SHM_NAME)
            .ok()?;

        let fd = file.into_raw_fd();
        let shm = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                std::mem::size_of::<NcclShm>(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            ) as *mut NcclShm
        };

        // Close the fd — mmap keeps its own reference
        unsafe { libc::close(fd); }

        if shm.is_null() || shm == libc::MAP_FAILED as *mut NcclShm {
            return None;
        }

        // Verify magic
        let magic = unsafe { (*shm).magic };
        if magic != COMM_MAGIC {
            unsafe { libc::munmap(shm as *mut libc::c_void, std::mem::size_of::<NcclShm>()); }
            return None;
        }

        // Start reading from current write position (skip old events)
        let read_idx = unsafe {
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            (*shm).write_idx
        };
        Some(Self { shm, read_idx })
    }

    pub fn poll(&mut self) -> Vec<NcclEvent> {
        let mut out = Vec::new();
        if self.shm.is_null() { return out; }

        let write_idx = unsafe {
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            (*self.shm).write_idx
        };

        while self.read_idx != write_idx {
            let idx = (self.read_idx as usize) % SHM_CAPACITY;
            let ev = unsafe { (*self.shm).events[idx] };
            out.push(ev);
            self.read_idx = self.read_idx.wrapping_add(1);
        }
        out
    }
}

impl Drop for NcclReader {
    fn drop(&mut self) {
        if !self.shm.is_null() {
            unsafe { libc::munmap(self.shm as *mut libc::c_void, std::mem::size_of::<NcclShm>()); }
        }
    }
}

pub fn event_type_str(t: u8) -> &'static str {
    match t {
        0 => "allreduce",
        1 => "allgather",
        2 => "reducescatter",
        3 => "broadcast",
        4 => "reduce",
        5 => "send",
        6 => "recv",
        _ => "unknown",
    }
}

pub fn datatype_str(t: u8) -> &'static str {
    match t {
        0 => "i8", 1 => "u8", 2 => "i32", 3 => "u32",
        4 => "i64", 5 => "u64", 6 => "f16", 7 => "f32",
        8 => "f64", 9 => "bf16", _ => "?",
    }
}
