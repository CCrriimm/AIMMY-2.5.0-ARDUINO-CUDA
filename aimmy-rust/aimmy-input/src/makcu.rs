use serialport::SerialPort;
use std::io::{self, Write};
use std::time::Duration;
use std::thread;

/// Makcu mouse backend (high baud rate, default 4,000,000).
pub struct MakcuInput {
    port: Box<dyn SerialPort>,
}

impl MakcuInput {
    pub fn open(com_port: &str, baud_rate: u32) -> io::Result<Self> {
        let port_number = extract_port_number(com_port);
        if port_number.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "Invalid COM port input."));
        }
        let com_port = format!("COM{port_number}");
        let port = serialport::new(&com_port, baud_rate)
            .timeout(Duration::from_millis(100))
            .open()?;
        thread::sleep(Duration::from_millis(100));
        Ok(Self { port })
    }

    pub fn send_click(&mut self, down: i32) -> io::Result<()> {
        // Makcu click protocol (implementation-specific).
        let command = format!("CLICK {down}\n");
        self.port.write_all(command.as_bytes())?;
        self.port.flush()
    }

    pub fn move_(&mut self, x: i32, y: i32) -> io::Result<()> {
        let command = format!("MOVE {x} {y}\n");
        self.port.write_all(command.as_bytes())?;
        self.port.flush()
    }

    /// The underlying `SerialPort` is closed when this struct is dropped.
    pub fn close(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn extract_port_number(com_port: &str) -> String {
    com_port.chars().filter(|c| c.is_ascii_digit()).collect()
}
