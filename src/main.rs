#[path = "bpf/rdma_probe.skel.rs"]
mod rdma_probe;
use rdma_probe::*;

mod app;
mod nccl;
mod tui;

use app::{App, NcclRecord, RdmaEvent};
use nccl::NcclReader;

use anyhow::{Context, Result};
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use libbpf_rs::skel::{OpenSkel, Skel, SkelBuilder};
use libbpf_rs::RingBufferBuilder;
use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Parser)]
#[command(name = "nccl-inspector", about = "Real-time NCCL & RDMA performance inspector")]
struct Cli {
    /// Disable RDMA tracing (skips eBPF, no root required)
    #[arg(long)]
    no_rdma: bool,

    /// Disable NCCL tracing
    #[arg(long)]
    no_nccl: bool,

    /// Output format: tui, json, plain
    #[arg(long, value_parser = ["tui", "json", "plain"], default_value = "tui")]
    format: String,

    /// Run for N seconds then exit with summary
    #[arg(long)]
    duration: Option<u64>,

    /// Max events to keep in scrollback
    #[arg(long, default_value_t = 500)]
    scrollback: usize,
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    if cli.no_rdma && cli.no_nccl {
        anyhow::bail!("cannot disable both --no-rdma and --no-nccl");
    }

    let mut app = App::new(cli.scrollback, cli.duration);
    let rdma_buffer = app.rdma_buffer.clone();

    // Load eBPF probes if RDMA tracing is enabled
    let ringbuf = if !cli.no_rdma {
        Some(setup_ebpf(&rdma_buffer).context(
            "failed to load eBPF probes — are you running as root?\n  \
             Use --no-rdma to skip RDMA tracing",
        )?)
    } else {
        None
    };

    let result = match cli.format.as_str() {
        "tui" => run_tui(&mut app, ringbuf.as_ref(), &cli),
        "json" | "plain" => run_headless(&mut app, ringbuf.as_ref(), &cli),
        _ => unreachable!(),
    };

    app.print_summary();
    result
}

// ── eBPF setup ──────────────────────────────────────────────────

fn setup_ebpf(
    rdma_buffer: &Arc<Mutex<Vec<RdmaEvent>>>,
) -> Result<libbpf_rs::RingBuffer<'static>> {
    // Leak allocations to give the ring buffer a 'static lifetime.
    // Fine for a long-running application — freed on process exit.
    let obj: &'static mut MaybeUninit<_> = Box::leak(Box::new(MaybeUninit::uninit()));
    let skel_builder = RdmaProbeSkelBuilder::default();
    let open_skel = skel_builder.open(obj)?;
    let mut skel = open_skel.load()?;
    skel.attach()?;

    let skel: &'static mut _ = Box::leak(Box::new(skel));

    let buf = rdma_buffer.clone();
    let mut builder = RingBufferBuilder::new();
    builder.add(&skel.maps.events, move |data: &[u8]| {
        if data.len() >= std::mem::size_of::<RdmaEvent>() {
            let e = unsafe { *(data.as_ptr() as *const RdmaEvent) };
            buf.lock().unwrap().push(e);
        }
        0
    })?;

    Ok(builder.build()?)
}

// ── TUI mode ────────────────────────────────────────────────────

fn run_tui(
    app: &mut App,
    ringbuf: Option<&libbpf_rs::RingBuffer<'static>>,
    cli: &Cli,
) -> Result<()> {
    // Restore terminal on panic
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        ratatui::restore();
        original_hook(info);
    }));

    let mut terminal = ratatui::init();
    let mut nccl_reader = open_nccl(cli);

    let result = loop {
        if expired(app, cli) {
            break Ok(());
        }

        // Keyboard input
        if event::poll(Duration::from_millis(10))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => break Ok(()),
                        KeyCode::Char('c')
                            if key.modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            break Ok(())
                        }
                        KeyCode::Tab => app.toggle_panel(),
                        KeyCode::Up => app.scroll_up(),
                        KeyCode::Down => app.scroll_down(),
                        _ => {}
                    }
                }
            }
        }

        // Poll eBPF ring buffer
        if let Some(rb) = ringbuf {
            rb.poll(Duration::from_millis(10))?;
            app.drain_rdma();
        }

        // Poll NCCL shared memory
        poll_nccl(app, &mut nccl_reader, cli);

        // Render frame
        terminal.draw(|f| tui::render(f, app))?;
    };

    ratatui::restore();
    result
}

// ── Headless modes (json / plain) ───────────────────────────────

fn run_headless(
    app: &mut App,
    ringbuf: Option<&libbpf_rs::RingBuffer<'static>>,
    cli: &Cli,
) -> Result<()> {
    if cli.no_rdma {
        eprintln!("Tracing NCCL (RDMA disabled)... Ctrl+C to stop");
    } else {
        eprintln!("Tracing RDMA + NCCL... Ctrl+C to stop");
    }

    let mut nccl_reader = open_nccl(cli);

    loop {
        if expired(app, cli) {
            break;
        }

        // Poll eBPF
        if let Some(rb) = ringbuf {
            rb.poll(Duration::from_millis(100))?;
            let events: Vec<RdmaEvent> = app.rdma_buffer.lock().unwrap().drain(..).collect();
            for e in events {
                print_rdma(&e, &cli.format);
                app.push_rdma(e);
            }
        } else {
            std::thread::sleep(Duration::from_millis(100));
        }

        // Poll NCCL
        if !cli.no_nccl {
            if nccl_reader.is_none() {
                nccl_reader = NcclReader::open();
            }
            if let Some(ref mut reader) = nccl_reader {
                for ev in reader.poll() {
                    app.push_nccl(ev);
                    if let Some(rec) = app.nccl_events.back() {
                        print_nccl(rec, &cli.format);
                    }
                }
            }
        }
    }

    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────

fn open_nccl(cli: &Cli) -> Option<NcclReader> {
    if cli.no_nccl { None } else { NcclReader::open() }
}

fn poll_nccl(app: &mut App, reader: &mut Option<NcclReader>, cli: &Cli) {
    if cli.no_nccl {
        return;
    }
    if reader.is_none() {
        *reader = NcclReader::open();
    }
    if let Some(ref mut r) = reader {
        for ev in r.poll() {
            app.push_nccl(ev);
        }
    }
}

fn expired(app: &App, cli: &Cli) -> bool {
    cli.duration
        .map(|d| app.start_time.elapsed() >= Duration::from_secs(d))
        .unwrap_or(false)
}

fn print_rdma(e: &RdmaEvent, format: &str) {
    let comm = std::str::from_utf8(&e.comm)
        .unwrap_or("?")
        .trim_end_matches('\0');

    match format {
        "json" => {
            let json = match e.event_type {
                0 => serde_json::json!({
                    "source": "rdma",
                    "type": "send",
                    "timestamp_ns": e.timestamp_ns,
                    "qp_num": e.qp_num,
                    "bytes": e.bytes,
                    "pid": e.pid,
                    "comm": comm,
                }),
                1 => serde_json::json!({
                    "source": "rdma",
                    "type": "completion",
                    "timestamp_ns": e.timestamp_ns,
                    "qp_num": e.qp_num,
                    "latency_ns": e.latency_ns,
                    "bytes": e.bytes,
                    "opcode": e.opcode,
                    "pid": e.pid,
                    "comm": comm,
                }),
                _ => return,
            };
            println!("{}", json);
        }
        _ => match e.event_type {
            0 => println!(
                "[SEND] ts={:>15}ns QP={:<6} bytes={:<8} pid={} comm={}",
                e.timestamp_ns, e.qp_num, e.bytes, e.pid, comm,
            ),
            1 => println!(
                "[COMP] ts={:>15}ns QP={:<6} latency={:<10}ns ({:.3}us)",
                e.timestamp_ns,
                e.qp_num,
                e.latency_ns,
                e.latency_ns as f64 / 1000.0,
            ),
            _ => {}
        },
    }
}

fn print_nccl(rec: &NcclRecord, format: &str) {
    let ev = &rec.event;
    match format {
        "json" => {
            let json = serde_json::json!({
                "source": "nccl",
                "type": nccl::event_type_str(ev.event_type),
                "timestamp_ns": ev.timestamp_ns,
                "duration_ns": ev.duration_ns,
                "rank": ev.rank,
                "nranks": ev.nranks,
                "count": ev.count,
                "datatype": nccl::datatype_str(ev.datatype),
                "peer": ev.peer,
                "pid": ev.pid,
                "rdma_sends": rec.rdma_sends,
                "rdma_comps": rec.rdma_comps,
                "rdma_bytes": rec.rdma_bytes,
            });
            println!("{}", json);
        }
        _ => {
            println!(
                "[NCCL] {:>14} rank={}/{} count={} dtype={} dur={:.3}us  rdma={}pkts/{}B",
                nccl::event_type_str(ev.event_type),
                ev.rank,
                ev.nranks,
                ev.count,
                nccl::datatype_str(ev.datatype),
                ev.duration_ns as f64 / 1000.0,
                rec.rdma_sends,
                rec.rdma_bytes,
            );
        }
    }
}
