
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::nccl::{self, NcclEvent};

const MAX_LATENCIES: usize = 10_000;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RdmaEvent {
    pub timestamp_ns: u64,
    pub latency_ns: u64,
    pub qp_num: u32,
    pub bytes: u32,
    pub opcode: u32,
    pub pid: u32,
    pub event_type: u8,
    pub comm: [u8; 16],
}

#[derive(Clone, Copy)]
pub struct NcclRecord {
    pub event: NcclEvent,
    pub rdma_sends: u32,
    pub rdma_comps: u32,
    pub rdma_bytes: u64,
}

#[derive(PartialEq, Clone, Copy)]
pub enum Panel {
    Nccl,
    Rdma,
}

pub struct LatencyStats {
    pub count: usize,
    pub avg: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
}

pub struct CorrelationSummary {
    pub op: String,
    pub nccl_count: u64,
    pub correlated: u64,
    pub total_rdma_sends: u64,
    pub avg_sends_per_call: f64,
    pub avg_bytes_per_call: f64,
}

pub struct App {
    pub rdma_events: VecDeque<RdmaEvent>,
    pub nccl_events: VecDeque<NcclRecord>,
    pub rdma_latencies: HashMap<u32, VecDeque<f64>>,
    pub nccl_latencies: HashMap<u8, VecDeque<f64>>,
    pub total_rdma_send: u64,
    pub total_rdma_comp: u64,
    pub total_nccl: HashMap<u8, u64>,
    pub total_rdma_bytes: u64,
    pub start_time: Instant,
    pub duration: Option<u64>,
    pub active_panel: Panel,
    pub nccl_scroll: usize,
    pub rdma_scroll: usize,
    pub scrollback: usize,
    pub rdma_buffer: Arc<Mutex<Vec<RdmaEvent>>>,
}

impl App {
    pub fn new(scrollback: usize, duration: Option<u64>) -> Self {
        Self {
            rdma_events: VecDeque::new(),
            nccl_events: VecDeque::new(),
            rdma_latencies: HashMap::new(),
            nccl_latencies: HashMap::new(),
            total_rdma_send: 0,
            total_rdma_comp: 0,
            total_nccl: HashMap::new(),
            total_rdma_bytes: 0,
            start_time: Instant::now(),
            duration,
            active_panel: Panel::Nccl,
            nccl_scroll: 0,
            rdma_scroll: 0,
            scrollback,
            rdma_buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn push_rdma(&mut self, event: RdmaEvent) {
        self.total_rdma_bytes += event.bytes as u64;
        match event.event_type {
            0 => self.total_rdma_send += 1,
            1 => {
                self.total_rdma_comp += 1;
                if event.latency_ns > 0 {
                    let bucket = self.rdma_latencies.entry(event.opcode).or_default();
                    if bucket.len() >= MAX_LATENCIES {
                        bucket.pop_front();
                    }
                    bucket.push_back(event.latency_ns as f64);
                }
            }
            _ => {}
        }
        if self.rdma_events.len() >= self.scrollback {
            self.rdma_events.pop_front();
        }
        self.rdma_events.push_back(event);
    }

    pub fn push_nccl(&mut self, event: NcclEvent) {
        *self.total_nccl.entry(event.event_type).or_insert(0) += 1;
        let latencies = self.nccl_latencies.entry(event.event_type).or_default();
        if latencies.len() >= MAX_LATENCIES {
            latencies.pop_front();
        }
        latencies.push_back(event.duration_ns as f64);

        let record = self.correlate(event);
        if self.nccl_events.len() >= self.scrollback {
            self.nccl_events.pop_front();
        }
        self.nccl_events.push_back(record);
    }

    /// Attach RDMA packet stats to an NCCL event by matching pid and
    /// the [start, start+duration] window. Both streams use CLOCK_MONOTONIC
    /// (shim via clock_gettime, eBPF via bpf_ktime_get_ns) so timestamps
    /// are directly comparable.
    fn correlate(&self, event: NcclEvent) -> NcclRecord {
        let mut record = NcclRecord {
            event,
            rdma_sends: 0,
            rdma_comps: 0,
            rdma_bytes: 0,
        };

        if event.pid == 0 {
            return record;
        }

        let start = event.timestamp_ns;
        let end = start.saturating_add(event.duration_ns);

        for r in &self.rdma_events {
            if r.pid != event.pid {
                continue;
            }
            if r.timestamp_ns < start || r.timestamp_ns > end {
                continue;
            }
            match r.event_type {
                0 => {
                    record.rdma_sends = record.rdma_sends.saturating_add(1);
                    record.rdma_bytes = record.rdma_bytes.saturating_add(r.bytes as u64);
                }
                1 => {
                    record.rdma_comps = record.rdma_comps.saturating_add(1);
                }
                _ => {}
            }
        }

        record
    }

    pub fn correlation_by_op(&self) -> Vec<CorrelationSummary> {
        use std::collections::BTreeMap;
        let mut agg: BTreeMap<u8, (u64, u64, u64, u64)> = BTreeMap::new();
        for rec in &self.nccl_events {
            let entry = agg.entry(rec.event.event_type).or_insert((0, 0, 0, 0));
            entry.0 += 1;
            if rec.rdma_sends > 0 || rec.rdma_comps > 0 {
                entry.1 += 1;
            }
            entry.2 += rec.rdma_sends as u64;
            entry.3 += rec.rdma_bytes;
        }
        agg.into_iter()
            .map(|(op, (count, correlated, sends, bytes))| {
                let denom = count.max(1) as f64;
                CorrelationSummary {
                    op: nccl::event_type_str(op).to_string(),
                    nccl_count: count,
                    correlated,
                    total_rdma_sends: sends,
                    avg_sends_per_call: sends as f64 / denom,
                    avg_bytes_per_call: bytes as f64 / denom,
                }
            })
            .collect()
    }

    pub fn drain_rdma(&mut self) {
        let events: Vec<RdmaEvent> = {
            let mut buf = self.rdma_buffer.lock().unwrap();
            buf.drain(..).collect()
        };
        for e in events {
            self.push_rdma(e);
        }
    }

    pub fn toggle_panel(&mut self) {
        self.active_panel = match self.active_panel {
            Panel::Nccl => Panel::Rdma,
            Panel::Rdma => Panel::Nccl,
        };
    }

    pub fn scroll_up(&mut self) {
        match self.active_panel {
            Panel::Nccl => self.nccl_scroll = self.nccl_scroll.saturating_add(1),
            Panel::Rdma => self.rdma_scroll = self.rdma_scroll.saturating_add(1),
        }
    }

    pub fn scroll_down(&mut self) {
        match self.active_panel {
            Panel::Nccl => self.nccl_scroll = self.nccl_scroll.saturating_sub(1),
            Panel::Rdma => self.rdma_scroll = self.rdma_scroll.saturating_sub(1),
        }
    }

    pub fn total_nccl_count(&self) -> u64 {
        self.total_nccl.values().sum()
    }

    pub fn rdma_stats_by_op(&self) -> Vec<(String, LatencyStats)> {
        let mut ops: Vec<_> = self.rdma_latencies.keys().copied().collect();
        ops.sort();
        ops.iter()
            .filter_map(|&op| {
                let latencies = self.rdma_latencies.get(&op)?;
                let stats = compute_stats(latencies);
                if stats.count == 0 {
                    return None;
                }
                Some((format!("rdma {}", rdma_opcode_str(op)), stats))
            })
            .collect()
    }

    pub fn nccl_stats_by_op(&self) -> Vec<(String, LatencyStats)> {
        let mut ops: Vec<_> = self.nccl_latencies.keys().copied().collect();
        ops.sort();
        ops.iter()
            .filter_map(|&op| {
                let latencies = self.nccl_latencies.get(&op)?;
                let stats = compute_stats(latencies);
                if stats.count == 0 {
                    return None;
                }
                Some((nccl::event_type_str(op).to_string(), stats))
            })
            .collect()
    }

    pub fn print_summary(&self) {
        let elapsed = self.start_time.elapsed();
        let secs = elapsed.as_secs_f64().max(0.001);

        eprintln!();
        eprintln!("=== NCCL Inspector Summary ===");
        eprintln!("Duration: {:.1}s", secs);
        eprintln!(
            "RDMA: {} sends, {} completions ({:.0} events/s)",
            self.total_rdma_send,
            self.total_rdma_comp,
            (self.total_rdma_send + self.total_rdma_comp) as f64 / secs,
        );
        eprintln!(
            "NCCL: {} events ({:.0} events/s)",
            self.total_nccl_count(),
            self.total_nccl_count() as f64 / secs,
        );
        eprintln!("Total RDMA bytes: {}", format_bytes(self.total_rdma_bytes));

        let nccl_stats = self.nccl_stats_by_op();
        let rdma_stats = self.rdma_stats_by_op();

        if !nccl_stats.is_empty() || !rdma_stats.is_empty() {
            eprintln!();
            eprintln!(
                "{:<16} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10}",
                "Operation", "Count", "Avg", "P50", "P95", "P99", "Max"
            );
            eprintln!("{}", "-".repeat(76));
        }

        for (name, stats) in &nccl_stats {
            eprintln!(
                "{:<16} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10}",
                name,
                stats.count,
                format_latency_ns(stats.avg),
                format_latency_ns(stats.p50),
                format_latency_ns(stats.p95),
                format_latency_ns(stats.p99),
                format_latency_ns(stats.max),
            );
        }

        for (name, stats) in &rdma_stats {
            eprintln!(
                "{:<16} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10}",
                name,
                stats.count,
                format_latency_ns(stats.avg),
                format_latency_ns(stats.p50),
                format_latency_ns(stats.p95),
                format_latency_ns(stats.p99),
                format_latency_ns(stats.max),
            );
        }

        let corr = self.correlation_by_op();
        if !corr.is_empty() {
            eprintln!();
            eprintln!("=== NCCL <-> RDMA correlation ===");
            eprintln!(
                "{:<16} {:>8} {:>10} {:>12} {:>12} {:>14}",
                "Operation", "Calls", "Matched", "RDMA pkts", "Avg pkts", "Avg bytes"
            );
            eprintln!("{}", "-".repeat(76));
            for c in &corr {
                eprintln!(
                    "{:<16} {:>8} {:>10} {:>12} {:>12.1} {:>14}",
                    c.op,
                    c.nccl_count,
                    c.correlated,
                    c.total_rdma_sends,
                    c.avg_sends_per_call,
                    format_bytes(c.avg_bytes_per_call as u64),
                );
            }
        }
    }
}

/// Maps `enum ib_wc_opcode` values to short labels. Values follow the
/// Linux UAPI (see include/rdma/ib_verbs.h).
pub fn rdma_opcode_str(op: u32) -> &'static str {
    match op {
        0 => "send",
        1 => "write",
        2 => "read",
        3 => "cmp_swap",
        4 => "fetch_add",
        5 => "lso",
        6 => "local_inv",
        7 => "reg_mr",
        8 => "masked_cmp_swap",
        9 => "masked_fetch_add",
        10 => "flush",
        11 => "atomic_write",
        128 => "recv",
        129 => "recv_imm",
        _ => "op?",
    }
}

fn compute_stats(latencies: &VecDeque<f64>) -> LatencyStats {
    if latencies.is_empty() {
        return LatencyStats {
            count: 0,
            avg: 0.0,
            p50: 0.0,
            p95: 0.0,
            p99: 0.0,
            max: 0.0,
        };
    }

    let mut sorted: Vec<f64> = latencies.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let count = sorted.len();
    let avg = sorted.iter().sum::<f64>() / count as f64;

    LatencyStats {
        count,
        avg,
        p50: percentile(&sorted, 50.0),
        p95: percentile(&sorted, 95.0),
        p99: percentile(&sorted, 99.0),
        max: sorted[count - 1],
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

pub fn format_latency_ns(ns: f64) -> String {
    if ns == 0.0 {
        "-".to_string()
    } else if ns < 1_000.0 {
        format!("{:.0}ns", ns)
    } else if ns < 1_000_000.0 {
        format!("{:.1}us", ns / 1_000.0)
    } else if ns < 1_000_000_000.0 {
        format!("{:.2}ms", ns / 1_000_000.0)
    } else {
        format!("{:.2}s", ns / 1_000_000_000.0)
    }
}

pub fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{}B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2}GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}
