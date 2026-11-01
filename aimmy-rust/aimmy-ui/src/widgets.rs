use aimmy_core::config::{AppConfig, ToggleState};
use eframe::egui;

/// A checkbox-style toggle switch with a label.
pub fn toggle(ui: &mut egui::Ui, config: &mut AppConfig, key: &str, label: &str) -> bool {
    let mut value = config.toggle_state.get(key);
    let changed = ui.checkbox(&mut value, label).changed();
    if changed {
        config.toggle_state.set(key, value);
    }
    value
}

/// A numeric slider.
pub fn slider(
    ui: &mut egui::Ui,
    config: &mut AppConfig,
    key: &str,
    label: &str,
    range: std::ops::RangeInclusive<f64>,
) -> bool {
    let mut value = config.slider_settings.get(key);
    let mut changed = false;
    ui.add(egui::Slider::new(&mut value, range).text(label)).on_hover_ui(|_| {});
    if value != config.slider_settings.get(key) {
        config.slider_settings.set(key, value);
        changed = true;
    }
    changed
}

/// A dropdown (combo box).
pub fn dropdown(
    ui: &mut egui::Ui,
    config: &mut AppConfig,
    key: &str,
    label: &str,
    options: &[&str],
) {
    let mut current = config.dropdown_state.get(key);
    egui::ComboBox::new(label, &mut current)
        .show_text(&current)
        .show_ui(ui, |ui| {
            for opt in options {
                if ui.selectable_value(&mut current, opt.to_string(), opt).clicked() {
                    config.dropdown_state.set(key, opt);
                }
            }
        });
    if current != config.dropdown_state.get(key) {
        config.dropdown_state.set(key, &current);
    }
}

/// A keybind picker: shows the bound key and enters "listening" mode on click.
pub struct KeyChanger {
    pub listening: bool,
}

impl KeyChanger {
    pub fn new() -> Self {
        Self { listening: false }
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        config: &mut AppConfig,
        key: &str,
        label: &str,
        bindings: &mut aimmy_bindings::InputBindingManager,
    ) {
        let mut current = config.binding_settings.get(key);
        ui.horizontal(|ui| {
            ui.label(label);
            if ui
                .button(if self.listening { "..." } else { &current })
                .clicked()
            {
                self.listening = !self.listening;
                if self.listening {
                    bindings.start_listening_for_binding(key);
                }
            }
        });
        // In a full build the hook feeds `bindings.on_key_down/on_key_up`;
        // here we expose the current binding for the UI thread to apply.
        if !self.listening && current != config.binding_settings.get(key) {
            config.binding_settings.set(key, &current);
        }
    }
}

/// A color picker button showing a color swatch.
pub fn color_changer(ui: &mut egui::Ui, config: &mut AppConfig, key: &str, label: &str) {
    let mut hex = config.color_state.get(key);
    let color = parse_hex(&hex).unwrap_or(egui::Color32::WHITE);
    ui.horizontal(|ui| {
        ui.label(label);
        let rect = egui::Rect::from_min_size(ui.cursor().min, egui::vec2(20.0, 20.0));
        ui.painter().rect_filled(rect, 4.0, color);
        if ui.button("pick").clicked() {
            // In a full build this opens a native color dialog.
        }
    });
    if hex != config.color_state.get(key) {
        config.color_state.set(key, &hex);
    }
}

fn parse_hex(hex: &str) -> Option<egui::Color32> {
    let s = hex.trim_start_matches('#');
    if s.len() != 8 {
        return None;
    }
    let a = u8::from_str_radix(&s[0..2], 16).ok()?;
    let r = u8::from_str_radix(&s[2..4], 16).ok()?;
    let g = u8::from_str_radix(&s[4..6], 16).ok()?;
    let b = u8::from_str_radix(&s[6..8], 16).ok()?;
    Some(egui::Color32::from_rgba_unmultiplied(r, g, b, a))
}

/// Minimizable section header.
pub struct Section {
    pub title: String,
    pub minimized: bool,
}

impl Section {
    pub fn show(ui: &mut egui::Ui, config: &mut AppConfig, minimize_key: &str, title: &str) -> bool {
        let mut minimized = config.minimize_state.get(minimize_key);
        let header = egui::CollapsingHeader::new(title)
            .id_source(minimize_key)
            .open(!minimized)
            .show(ui);
        let body_open = header.body_open();
        if minimized != !body_open {
            config.minimize_state.set(minimize_key, !body_open);
        }
        body_open
    }
}
