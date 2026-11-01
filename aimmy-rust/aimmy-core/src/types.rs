use serde::{Deserialize, Serialize};

/// A rectangle with floating point coordinates (port of System.Drawing.RectangleF).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RectangleF {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl RectangleF {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    pub fn left(&self) -> f32 { self.x }
    pub fn top(&self) -> f32 { self.y }
    pub fn right(&self) -> f32 { self.x + self.width }
    pub fn bottom(&self) -> f32 { self.y + self.height }
    pub fn center_x(&self) -> f32 { self.x + self.width / 2.0 }
    pub fn center_y(&self) -> f32 { self.y + self.height / 2.0 }

    /// Returns true if `point` is inside the rectangle (inclusive).
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.right() && py >= self.y && py <= self.bottom()
    }
}

/// Integer rectangle (port of System.Drawing.Rectangle).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RectangleI {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

impl RectangleI {
    pub fn new(left: i32, top: i32, right: i32, bottom: i32) -> Self {
        Self { left, top, right, bottom }
    }

    pub fn from_xywh(x: i32, y: i32, w: i32, h: i32) -> Self {
        Self { left: x, top: y, right: x + w, bottom: y + h }
    }

    pub fn width(&self) -> i32 { self.right - self.left }
    pub fn height(&self) -> i32 { self.bottom - self.top }
    pub fn center_x(&self) -> i32 { self.left + self.width() / 2 }
    pub fn center_y(&self) -> i32 { self.top + self.height() / 2 }
}

/// A detected player/target, port of Aimmy's `Prediction` class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub rectangle: RectangleF,
    pub confidence: f32,
    pub class_id: i32,
    pub class_name: String,
    /// Normalised centre coordinates within the capture image [0..1].
    pub center_x_translated: f32,
    pub center_y_translated: f32,
    /// Absolute screen-space position of the target centre.
    pub screen_center_x: f32,
    pub screen_center_y: f32,
}

impl Default for Prediction {
    fn default() -> Self {
        Self {
            rectangle: RectangleF::new(0.0, 0.0, 0.0, 0.0),
            confidence: 0.0,
            class_id: 0,
            class_name: "Enemy".into(),
            center_x_translated: 0.0,
            center_y_translated: 0.0,
            screen_center_x: 0.0,
            screen_center_y: 0.0,
        }
    }
}

/// Information about a physical display/monitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayInfo {
    pub index: i32,
    pub is_primary: bool,
    pub device_name: String,
    /// Monitor bounds in virtual screen coordinates.
    pub bounds: RectangleI,
    /// Working area (excludes taskbar).
    pub working_area: RectangleI,
}

impl DisplayInfo {
    pub fn contains(&self, px: i32, py: i32) -> bool {
        px >= self.bounds.left && px <= self.bounds.right && py >= self.bounds.top && py <= self.bounds.bottom
    }
}

/// 2D point.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PointF {
    pub x: f32,
    pub y: f32,
}

impl PointF {
    pub fn new(x: f32, y: f32) -> Self { Self { x, y } }
}
