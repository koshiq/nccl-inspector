use ratatui::prelude::*;
use ratatui::widgets::{Block, BorderType, Cell, Paragraph, Row, Table};

use crate::app::{format_latency_ns, App, Panel};
use crate::nccl;

pub fn render(f: &mut Frame, app: &App) {
    let chunks = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(8),
        Constraint::Length(12),
        Constraint::Length(1),
    ])
    .split(f.area());

    render_header(f, app, chunks[0]);
    render_events(f, app, chunks[1]);
    render_stats(f, app, chunks[2]);
    render_footer(f, chunks[3]);
}

fn render_header(f: &mut Frame, app: &App, area: Rect) {
    let elapsed = app.start_time.elapsed();
    let uptime = format!(
        "{:02}:{:02}:{:02}",
        elapsed.as_secs() / 3600,
        (elapsed.as_secs() % 3600) / 60,
        elapsed.as_secs() % 60,
    );

    let rdma_total = app.total_rdma_send + app.total_rdma_comp;
    let nccl_total = app.total_nccl_count();

    let mut spans = vec![
        Span::styled(" RDMA ", Style::default().fg(Color::Yellow).bold()),
        Span::raw(format!("{} events", rdma_total)),
        Span::styled("  |  ", Style::default().fg(Color::DarkGray)),
        Span::styled(" NCCL ", Style::default().fg(Color::Green).bold()),
        Span::raw(format!("{} events", nccl_total)),
        Span::styled("  |  ", Style::default().fg(Color::DarkGray)),
        Span::styled(" Up ", Style::default().fg(Color::Cyan).bold()),
        Span::raw(uptime),
    ];

    if let Some(d) = app.duration {
        let remaining = d.saturating_sub(elapsed.as_secs());
        spans.push(Span::styled("  |  ", Style::default().fg(Color::DarkGray)));
        spans.push(Span::styled(
            format!(" {}s left ", remaining),
            Style::default().fg(Color::Red).bold(),
        ));
    }

    let header = Paragraph::new(Line::from(spans)).block(
        Block::bordered()
            .border_type(BorderType::Rounded)
            .title(" nccl-inspector ")
            .title_style(Style::default().fg(Color::Cyan).bold())
            .border_style(Style::default().fg(Color::Cyan)),
    );

    f.render_widget(header, area);
}

fn render_events(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::horizontal([
        Constraint::Percentage(50),
        Constraint::Percentage(50),
    ])
    .split(area);

    render_nccl_table(f, app, chunks[0]);
    render_rdma_table(f, app, chunks[1]);
}

fn render_nccl_table(f: &mut Frame, app: &App, area: Rect) {
    let is_active = app.active_panel == Panel::Nccl;
    let border_color = if is_active { Color::Cyan } else { Color::DarkGray };
    let title_style = if is_active {
        Style::default().fg(Color::Green).bold()
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .title(" NCCL Collectives ")
        .title_style(title_style)
        .border_style(Style::default().fg(border_color));

    let visible = (area.height as usize).saturating_sub(4); // borders + header row
    let total = app.nccl_events.len();
    let scroll = app.nccl_scroll.min(total.saturating_sub(visible));
    let start = total.saturating_sub(visible + scroll);

    let rows: Vec<Row> = app
        .nccl_events
        .iter()
        .skip(start)
        .take(visible)
        .map(|rec| {
            let ev = &rec.event;
            let op = nccl::event_type_str(ev.event_type);
            let rank = format!("{}/{}", ev.rank, ev.nranks);
            let count = format_count(ev.count);
            let dtype = nccl::datatype_str(ev.datatype);
            let dur = format_latency_ns(ev.duration_ns as f64);
            let (pkts_text, pkts_style) = if rec.rdma_sends == 0 {
                ("-".to_string(), Style::default().fg(Color::DarkGray))
            } else {
                (
                    format!("{}", rec.rdma_sends),
                    Style::default().fg(Color::Yellow),
                )
            };
            let bytes_text = if rec.rdma_bytes == 0 {
                "-".to_string()
            } else {
                format_count(rec.rdma_bytes)
            };

            Row::new(vec![
                Cell::from(op).style(Style::default().fg(Color::Green)),
                Cell::from(rank),
                Cell::from(count),
                Cell::from(dtype),
                Cell::from(dur).style(Style::default().fg(Color::White)),
                Cell::from(pkts_text).style(pkts_style),
                Cell::from(bytes_text).style(Style::default().fg(Color::Magenta)),
            ])
        })
        .collect();

    let header = Row::new(vec!["Op", "Rank", "Count", "Dtype", "Duration", "Pkts", "RdmaB"])
        .style(Style::default().fg(Color::White).bold());

    let widths = [
        Constraint::Length(10),
        Constraint::Length(6),
        Constraint::Length(7),
        Constraint::Length(5),
        Constraint::Length(9),
        Constraint::Length(5),
        Constraint::Length(7),
    ];

    let table = Table::new(rows, widths).header(header).block(block);
    f.render_widget(table, area);
}

fn render_rdma_table(f: &mut Frame, app: &App, area: Rect) {
    let is_active = app.active_panel == Panel::Rdma;
    let border_color = if is_active { Color::Cyan } else { Color::DarkGray };
    let title_style = if is_active {
        Style::default().fg(Color::Yellow).bold()
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .title(" RDMA Events ")
        .title_style(title_style)
        .border_style(Style::default().fg(border_color));

    let visible = (area.height as usize).saturating_sub(4);
    let total = app.rdma_events.len();
    let scroll = app.rdma_scroll.min(total.saturating_sub(visible));
    let start = total.saturating_sub(visible + scroll);

    let rows: Vec<Row> = app
        .rdma_events
        .iter()
        .skip(start)
        .take(visible)
        .map(|ev| {
            let (etype, color) = match ev.event_type {
                0 => ("SEND", Color::Yellow),
                1 => ("COMP", Color::Magenta),
                _ => ("?", Color::White),
            };
            let qp = format!("{}", ev.qp_num);
            let bytes = format_count(ev.bytes as u64);
            let latency = if ev.event_type == 1 && ev.latency_ns > 0 {
                format_latency_ns(ev.latency_ns as f64)
            } else {
                "-".to_string()
            };
            let comm = std::str::from_utf8(&ev.comm)
                .unwrap_or("?")
                .trim_end_matches('\0');

            Row::new(vec![
                Cell::from(etype).style(Style::default().fg(color)),
                Cell::from(qp),
                Cell::from(bytes),
                Cell::from(latency),
                Cell::from(comm.to_string()).style(Style::default().fg(Color::DarkGray)),
            ])
        })
        .collect();

    let header = Row::new(vec!["Type", "QP", "Bytes", "Latency", "Comm"])
        .style(Style::default().fg(Color::White).bold());

    let widths = [
        Constraint::Length(6),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(10),
        Constraint::Length(10),
    ];

    let table = Table::new(rows, widths).header(header).block(block);
    f.render_widget(table, area);
}

fn render_stats(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .title(" Latency Statistics ")
        .title_style(Style::default().fg(Color::White).bold())
        .border_style(Style::default().fg(Color::DarkGray));

    let mut rows: Vec<Row> = Vec::new();

    for (name, stats) in app.nccl_stats_by_op() {
        rows.push(Row::new(vec![
            Cell::from(name).style(Style::default().fg(Color::Green)),
            Cell::from(format!("{}", stats.count)),
            Cell::from(format_latency_ns(stats.avg)),
            Cell::from(format_latency_ns(stats.p50)),
            Cell::from(format_latency_ns(stats.p95)).style(latency_color(stats.p95, stats.avg)),
            Cell::from(format_latency_ns(stats.p99)).style(latency_color(stats.p99, stats.avg)),
            Cell::from(format_latency_ns(stats.max)).style(latency_color(stats.max, stats.avg)),
        ]));
    }

    for (name, stats) in app.rdma_stats_by_op() {
        rows.push(Row::new(vec![
            Cell::from(name).style(Style::default().fg(Color::Yellow)),
            Cell::from(format!("{}", stats.count)),
            Cell::from(format_latency_ns(stats.avg)),
            Cell::from(format_latency_ns(stats.p50)),
            Cell::from(format_latency_ns(stats.p95)).style(latency_color(stats.p95, stats.avg)),
            Cell::from(format_latency_ns(stats.p99)).style(latency_color(stats.p99, stats.avg)),
            Cell::from(format_latency_ns(stats.max)).style(latency_color(stats.max, stats.avg)),
        ]));
    }

    if rows.is_empty() {
        rows.push(Row::new(vec![Cell::from(
            "  Waiting for events...",
        )
        .style(Style::default().fg(Color::DarkGray))]));
    }

    let header = Row::new(vec![
        "Operation", "Count", "Avg", "P50", "P95", "P99", "Max",
    ])
    .style(Style::default().fg(Color::White).bold());

    let widths = [
        Constraint::Length(14),
        Constraint::Length(8),
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Length(10),
    ];

    let table = Table::new(rows, widths).header(header).block(block);
    f.render_widget(table, area);
}

fn render_footer(f: &mut Frame, area: Rect) {
    let footer = Paragraph::new(Line::from(vec![
        Span::styled(" q ", Style::default().fg(Color::Cyan).bold()),
        Span::raw("Quit  "),
        Span::styled("Tab ", Style::default().fg(Color::Cyan).bold()),
        Span::raw("Switch Panel  "),
        Span::styled("Up/Down ", Style::default().fg(Color::Cyan).bold()),
        Span::raw("Scroll"),
    ]))
    .style(Style::default().fg(Color::DarkGray));

    f.render_widget(footer, area);
}

fn latency_color(value: f64, avg: f64) -> Style {
    if avg == 0.0 {
        return Style::default();
    }
    if value > avg * 3.0 {
        Style::default().fg(Color::Red)
    } else if value > avg * 2.0 {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default()
    }
}

fn format_count(n: u64) -> String {
    if n < 1_000 {
        format!("{}", n)
    } else if n < 1_000_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else if n < 1_000_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else {
        format!("{:.1}G", n as f64 / 1_000_000_000.0)
    }
}
