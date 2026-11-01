use chrono::Local;
use std::fs::{OpenOptions, File};
use std::io::{Write, BufWriter};
use std::path::Path;
use std::sync::Mutex;

/// Log severity levels, mirroring the C# `LogManager.LogLevel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Info,
    Warning,
    Error,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warning => write!(f, "WARNING"),
            LogLevel::Error => write!(f, "ERROR"),
        }
    }
}

/// Static-ish logger. Buffers writes to `bin/aimmy.log` and also prints to
/// stdout/stderr so console builds stay visible.
pub struct LogManager {
    writer: Option<BufWriter<File>>,
    path: Option<String>,
}

static LOGGER: Mutex<Option<LogManager>> = Mutex::new(None);

impl LogManager {
    /// Initialise the logger. Safe to call multiple times.
    pub fn init<P: AsRef<Path>>(log_dir: P, file_name: &str) {
        let path = log_dir.as_ref().join(file_name);
        let _ = std::fs::create_dir_all(log_dir);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .ok()
            .map(BufWriter::new);
        let mut guard = LOGGER.lock().unwrap();
        *guard = Some(LogManager {
            writer: file,
            path: path.to_str().map(|s| s.to_string()),
        });
    }

    pub fn log(level: LogLevel, message: &str, notify: bool, duration_ms: u64) {
        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%%.3f").to_string();
        let line = format!("[{} {:<7}] {}", timestamp, level, message);

        // Console output
        match level {
            LogLevel::Error => eprintln!("{}", line),
            _ => println!("{}", line),
        }

        // File output
        let mut guard = LOGGER.lock().unwrap();
        if let Some(ref mut mgr) = *guard {
            if let Some(ref mut w) = mgr.writer {
                let _ = writeln!(w, "{}", line);
                let _ = w.flush();
            }
        }

        // In a GUI build this is where a transient notice bar would be shown.
        // The `notify` / `duration_ms` params are kept for API compatibility;
        // the UI crate handles toast display.
        let _ = (notify, duration_ms);
    }

    /// Convenience: info log.
    pub fn info(message: &str) {
        Self::log(LogLevel::Info, message, false, 0);
    }

    /// Convenience: warning log.
    pub fn warn(message: &str) {
        Self::log(LogLevel::Warning, message, true, 3000);
    }

    /// Convenience: error log.
    pub fn error(message: &str) {
        Self::log(LogLevel::Error, message, true, 6000);
    }
}
