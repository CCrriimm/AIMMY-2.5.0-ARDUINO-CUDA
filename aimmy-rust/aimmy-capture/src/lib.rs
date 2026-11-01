pub mod display;
pub mod capture;

pub use display::{DisplayManager, DisplayInfo, DisplayChangedEvent};
pub use capture::{CaptureManager, CaptureMethod, ScreenGrabResult};
