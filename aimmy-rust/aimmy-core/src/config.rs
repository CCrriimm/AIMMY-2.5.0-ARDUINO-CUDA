use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Central, serializable application state.
///
/// This is the Rust equivalent of the C# `Dictionary` static class. Every
/// user-facing setting lives here so it can be saved to / loaded from the
/// `bin/*.cfg` JSON files exactly as the original Aimmy does.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AppConfig {
    pub slider_settings: SliderSettings,
    pub toggle_state: ToggleState,
    pub dropdown_state: DropdownState,
    pub binding_settings: BindingSettings,
    pub color_state: ColorState,
    pub file_location_state: FileLocationState,
    pub minimize_state: MinimizeState,
}

impl AppConfig {
    /// Paths (relative to the executable directory) that Aimmy persists.
    pub const MODELS_DIR: &str = "bin/models";
    pub const CONFIGS_DIR: &str = "bin/configs";
    pub const IMAGES_DIR: &str = "bin/images";
    pub const LABELS_DIR: &str = "bin/labels";

    /// Persist the whole config as a single JSON object.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
        })?;
        std::fs::write(path, json)
    }

    /// Load the whole config from a JSON file, merging defaults for missing keys.
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let mut cfg: Self = serde_json::from_str(&data).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
        })?;
        cfg.apply_defaults();
        Ok(cfg)
    }

    /// Ensure every setting has a value (missing keys fall back to defaults).
    pub fn apply_defaults(&mut self) {
        let defaults = Self::default();
        if self.slider_settings.is_empty() {
            self.slider_settings = defaults.slider_settings;
        }
        if self.toggle_state.is_empty() {
            self.toggle_state = defaults.toggle_state;
        }
        if self.dropdown_state.is_empty() {
            self.dropdown_state = defaults.dropdown_state;
        }
        if self.binding_settings.is_empty() {
            self.binding_settings = defaults.binding_settings;
        }
        if self.color_state.is_empty() {
            self.color_state = defaults.color_state;
        }
        if self.file_location_state.is_empty() {
            self.file_location_state = defaults.file_location_state;
        }
        if self.minimize_state.is_empty() {
            self.minimize_state = defaults.minimize_state;
        }
    }
}

// ---------------------------------------------------------------------------
// Slider settings (double-precision numeric values)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SliderSettings(pub HashMap<String, f64>);

impl SliderSettings {
    pub fn new() -> Self {
        let mut m = HashMap::new();
        m.insert("Suggested Model".into(), 0.0);
        m.insert("FOV Size".into(), 640.0);
        m.insert("FOV Action Size".into(), 135.0);
        m.insert("Dynamic FOV Size".into(), 200.0);
        m.insert("Mouse Sensitivity (+/-)".into(), 0.80);
        m.insert("Mouse Jitter".into(), 4.0);
        m.insert("Sticky Aim Threshold".into(), 50.0);
        m.insert("Insta-Flick Speed".into(), 0.16);
        m.insert("Flick Aim Strength".into(), 0.37);
        m.insert("Prediction Pixel Offset".into(), 36.5);
        m.insert("Y Offset (Up/Down)".into(), 0.0);
        m.insert("Y Offset (%)".into(), 50.0);
        m.insert("X Offset (Left/Right)".into(), 0.0);
        m.insert("X Offset (%)".into(), 50.0);
        m.insert("EMA Smoothening".into(), 0.37);
        m.insert("Accelerated Smoothing".into(), 1.0);
        m.insert("Overshoot Damping".into(), 0.5);
        m.insert("Overshoot Recovery".into(), 0.35);
        m.insert("Kalman Lead Time".into(), 0.10);
        m.insert("WiseTheFox Lead Time".into(), 0.15);
        m.insert("Shalloe Lead Multiplier".into(), 3.0);
        m.insert("Auto Trigger Delay".into(), 0.1);
        m.insert("AI Minimum Confidence".into(), 45.0);
        m.insert("AI Confidence Font Size".into(), 20.0);
        m.insert("Corner Radius".into(), 0.0);
        m.insert("Border Thickness".into(), 1.0);
        m.insert("Opacity".into(), 1.0);
        m.insert("Selected Display".into(), 0.0);
        Self(m)
    }

    pub fn get(&self, key: &str) -> f64 {
        self.0.get(key).copied().unwrap_or(0.0)
    }

    pub fn set(&mut self, key: &str, value: f64) {
        self.0.insert(key.to_string(), value);
    }
}

// ---------------------------------------------------------------------------
// Toggle states (booleans)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToggleState(pub HashMap<String, bool>);

impl ToggleState {
    pub fn new() -> Self {
        let mut m = HashMap::new();
        m.insert("Aim Assist".into(), false);
        m.insert("Sticky Aim".into(), false);
        m.insert("Constant AI Tracking".into(), false);
        m.insert("Insta-Flick".into(), false);
        m.insert("Predictions".into(), false);
        m.insert("Flick Aim".into(), false);
        m.insert("Prediction Pixel Offset Enabled".into(), false);
        m.insert("EMA Smoothening".into(), false);
        m.insert("Accelerated Smoothing".into(), false);
        m.insert("Adaptive Sensitivity Control".into(), true);
        m.insert("Enable Model Switch Keybind".into(), true);
        m.insert("Auto Trigger".into(), false);
        m.insert("FOV".into(), false);
        m.insert("FOV Action".into(), false);
        m.insert("Dynamic FOV".into(), false);
        m.insert("Third Person Support".into(), false);
        m.insert("Masking".into(), false);
        m.insert("Show Detected Player".into(), false);
        m.insert("Cursor Check".into(), false);
        m.insert("Spray Mode".into(), false);
        m.insert("Show FOV".into(), true);
        m.insert("Show AI Confidence".into(), false);
        m.insert("Show Tracers".into(), false);
        m.insert("Collect Data While Playing".into(), false);
        m.insert("Auto Label Data".into(), false);
        m.insert("LG HUB Mouse Movement".into(), false);
        m.insert("Mouse Background Effect".into(), true);
        m.insert("Debug Mode".into(), false);
        m.insert("UI TopMost".into(), false);
        m.insert("StreamGuard".into(), false);
        m.insert("Realtime GPU Priority".into(), true);
        m.insert("X Axis Percentage Adjustment".into(), false);
        m.insert("Y Axis Percentage Adjustment".into(), false);
        Self(m)
    }

    pub fn get(&self, key: &str) -> bool {
        self.0.get(key).copied().unwrap_or(false)
    }

    pub fn set(&mut self, key: &str, value: bool) {
        self.0.insert(key.to_string(), value);
    }
}

// ---------------------------------------------------------------------------
// Dropdown states (strings)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DropdownState(pub HashMap<String, String>);

impl DropdownState {
    pub fn new() -> Self {
        let mut m = HashMap::new();
        m.insert("Prediction Method".into(), "Kalman Filter".into());
        m.insert("Detection Area Type".into(), "Closest to Center Screen".into());
        m.insert("Aiming Boundaries Alignment".into(), "Center".into());
        m.insert("Mouse Movement Method".into(), "Arduino".into());
        m.insert("Arduino COM Port".into(), "".into());
        m.insert("Makcu Baud Rate".into(), "4000000".into());
        m.insert("Screen Capture Method".into(), "DirectX".into());
        m.insert("Execution Provider".into(), "".into());
        m.insert("Tracer Position".into(), "Bottom".into());
        m.insert("Movement Path".into(), "Cubic Bezier".into());
        m.insert("Image Size".into(), "640".into());
        m.insert("Target Class".into(), "Best Confidence".into());
        m.insert("FOV Style".into(), "Circle".into());
        Self(m)
    }

    pub fn get(&self, key: &str) -> String {
        self.0
            .get(key)
            .cloned()
            .unwrap_or_else(|| String::new())
    }

    pub fn set(&mut self, key: &str, value: &str) {
        self.0.insert(key.to_string(), value.to_string());
    }
}

// ---------------------------------------------------------------------------
// Keybind settings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BindingSettings(pub HashMap<String, String>);

impl BindingSettings {
    pub fn new() -> Self {
        let mut m = HashMap::new();
        m.insert("Aim Keybind".into(), "Right".into());
        m.insert("Second Aim Keybind".into(), "LMenu".into());
        m.insert("Toggle Aim Keybind".into(), "None".into());
        m.insert("Insta-Flick Keybind".into(), "OemQuotes".into());
        m.insert("Second Insta-Flick Keybind".into(), "None".into());
        m.insert("Dynamic FOV Keybind".into(), "Left".into());
        m.insert("Emergency Stop Keybind".into(), "Delete".into());
        m.insert("Model Switch Keybind".into(), "OemPipe".into());
        Self(m)
    }

    pub fn get(&self, key: &str) -> String {
        self.0.get(key).cloned().unwrap_or_default()
    }

    pub fn set(&mut self, key: &str, value: &str) {
        self.0.insert(key.to_string(), value.to_string());
    }
}

// ---------------------------------------------------------------------------
// Color states (hex strings like #AARRGGBB)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ColorState(pub HashMap<String, String>);

impl ColorState {
    pub fn new() -> Self {
        let mut m = HashMap::new();
        m.insert("FOV Color".into(), "#FF8080FF".into());
        m.insert("FOV Action Color".into(), "#FFFF2040".into());
        m.insert("Detected Player Color".into(), "#FF00FFFF".into());
        m.insert("Theme Color".into(), "#FF722ED1".into());
        Self(m)
    }

    pub fn get(&self, key: &str) -> String {
        self.0.get(key).cloned().unwrap_or_default()
    }

    pub fn set(&mut self, key: &str, value: &str) {
        self.0.insert(key.to_string(), value.to_string());
    }
}

// ---------------------------------------------------------------------------
// File location state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FileLocationState(pub HashMap<String, String>);

impl FileLocationState {
    pub fn new() -> Self {
        let mut m = HashMap::new();
        m.insert("ddxoft DLL Location".into(), "".into());
        Self(m)
    }

    pub fn get(&self, key: &str) -> String {
        self.0.get(key).cloned().unwrap_or_default()
    }

    pub fn set(&mut self, key: &str, value: &str) {
        self.0.insert(key.to_string(), value.to_string());
    }
}

// ---------------------------------------------------------------------------
// Minimize (panel collapse) state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MinimizeState(pub HashMap<String, bool>);

impl MinimizeState {
    pub fn new() -> Self {
        let mut m = HashMap::new();
        m.insert("Aim Assist".into(), false);
        m.insert("Aim Config".into(), false);
        m.insert("Predictions".into(), false);
        m.insert("Auto Trigger".into(), false);
        m.insert("FOV Config".into(), false);
        m.insert("ESP Config".into(), false);
        m.insert("Model Settings".into(), false);
        m.insert("Settings Menu".into(), false);
        m.insert("X/Y Percentage Adjustment".into(), false);
        m.insert("Theme Settings".into(), false);
        m.insert("Screen Settings".into(), false);
        Self(m)
    }

    pub fn get(&self, key: &str) -> bool {
        self.0.get(key).copied().unwrap_or(false)
    }

    pub fn set(&mut self, key: &str, value: bool) {
        self.0.insert(key.to_string(), value);
    }
}
