use serialport::SerialPort;
use std::io::{self, Write};
use std::time::Duration;
use std::thread;

/// Arduino mouse backend: sends `{x},{y},{click}\n` over serial at 115200 baud.
/// Port of Aimmy's `ArduinoSupport.ArduinoInput`.
pub struct ArduinoInput {
    port: Box<dyn SerialPort>,
}

impl ArduinoInput {
    pub fn open(com_port: &str) -> io::Result<Self> {
        let port_number = extract_port_number(com_port);
        if port_number.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "Invalid COM port input."));
        }
        let com_port = format!("COM{port_number}");
        let mut port = serialport::new(&com_port, 115200)
            .timeout(Duration::from_millis(100))
            .open()?;
        thread::sleep(Duration::from_millis(200)); // let the Arduino reset
        Ok(Self { port })
    }

    pub fn send_mouse_command(&mut self, x: i32, y: i32, click: i32) -> io::Result<()> {
        let command = format!("{x},{y},{click}\n");
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
