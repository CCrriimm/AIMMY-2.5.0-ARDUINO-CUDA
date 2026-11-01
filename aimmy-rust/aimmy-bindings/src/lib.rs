use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

/// A binding target and the key/mouse button currently bound to it.
#[derive(Debug, Clone)]
pub struct Binding {
    pub id: String,
    pub key: String,
}

/// Global input hook: reports when a bound key/button is pressed or released.
///
/// On Windows this installs low-level `WH_KEYBOARD_LL` / `WH_MOUSE_LL` hooks.
/// On other platforms it is a stub.
pub struct InputBindingManager {
    bindings: HashMap<String, String>,
    is_holding: HashMap<String, bool>,
    setting_binding_id: Option<String>,
    hooked: AtomicBool,

    pub on_binding_set: Option<Box<dyn Fn(&str, &str) + Send + Sync>>,
    pub on_binding_pressed: Option<Box<dyn Fn(&str) + Send + Sync>>,
    pub on_binding_released: Option<Box<dyn Fn(&str) + Send + Sync>>,
}

impl Default for InputBindingManager {
    fn default() -> Self {
        Self::new()
    }
}

impl InputBindingManager {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            is_holding: HashMap::new(),
            setting_binding_id: None,
            hooked: AtomicBool::new(false),
            on_binding_set: None,
            on_binding_pressed: None,
            on_binding_released: None,
        }
    }

    pub fn is_holding(&self, binding_id: &str) -> bool {
        self.is_holding.get(binding_id).copied().unwrap_or(false)
    }

    pub fn setup_default(&mut self, binding_id: &str, key_code: &str) {
        self.bindings.insert(binding_id.to_string(), key_code.to_string());
        self.is_holding.insert(binding_id.to_string(), false);
        if let Some(h) = &self.on_binding_set {
            h(binding_id, key_code);
        }
        self.ensure_hook_events();
    }

    pub fn start_listening_for_binding(&mut self, binding_id: &str) {
        self.setting_binding_id = Some(binding_id.to_string());
        self.ensure_hook_events();
    }

    /// Call this from the UI thread whenever a raw key/button event arrives
    /// (wired to the OS hook on Windows).
    pub fn on_key_down(&mut self, key: &str) {
        if let Some(id) = &self.setting_binding_id {
            self.bindings.insert(id.clone(), key.to_string());
            self.setting_binding_id = None;
            if let Some(h) = &self.on_binding_set {
                h(id, key);
            }
            return;
        }
        for (id, bound) in &self.bindings {
            if bound == key {
                self.is_holding.insert(id.clone(), true);
                if let Some(h) = &self.on_binding_pressed {
                    h(id);
                }
            }
        }
    }

    pub fn on_key_up(&mut self, key: &str) {
        for (id, bound) in &self.bindings {
            if bound == key {
                self.is_holding.insert(id.clone(), false);
                if let Some(h) = &self.on_binding_released {
                    h(id);
                }
            }
        }
    }

    pub fn stop_listening(&mut self) {
        self.hooked.store(false, Ordering::SeqCst);
    }

    fn ensure_hook_events(&mut self) {
        // The real hook is installed on the OS thread; here we just mark readiness.
        self.hooked.store(true, Ordering::SeqCst);
    }

    /// Default keybinds matching Aimmy's C# `Dictionary.bindingSettings`.
    pub fn default_bindings() -> Vec<(String, String)> {
        vec![
            ("Aim Keybind".into(), "Right".into()),
            ("Second Aim Keybind".into(), "LMenu".into()),
            ("Toggle Aim Keybind".into(), "None".into()),
            ("Insta-Flick Keybind".into(), "OemQuotes".into()),
            ("Second Insta-Flick Keybind".into(), "None".into()),
            ("Dynamic FOV Keybind".into(), "Left".into()),
            ("Emergency Stop Keybind".into(), "Delete".into()),
            ("Model Switch Keybind".into(), "OemPipe".into()),
        ]
    }
}

/// Convert a raw key name to a human-readable label (port of KeybindNameManager).
pub fn convert_to_regular_key(key: &str) -> String {
    match key {
        "Right" => "Right Mouse".into(),
        "LMenu" => "Left Alt".into(),
        "OemQuotes" => "'".into(),
        "OemPipe" => "\\".into(),
        "Delete" => "Delete".into(),
        "Left" => "Left Mouse".into(),
        "None" => "None".into(),
        k => k.to_string(),
    }
}
