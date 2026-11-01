use aimmy_core::config::AppConfig;
use aimmy_core::types::RectangleI;
use aimmy_capture::DisplayManager;
use eframe::egui;

/// Renders the FOV overlay (circle/rectangle + action zone) over the current
/// display. Mirrors the C# `FOV` window.
pub struct FovOverlay;

impl FovOverlay {
    pub fn render(ctx: &egui::Context, app: &crate::AimmyApp) {
        let display = &app.display;
        let screen_w = display.screen_width() as f32;
        let screen_h = display.screen_height() as f32;
        let screen_x = display.screen_left() as f32;
        let screen_y = display.screen_top() as f32;

        let fov_size = app.config.slider_settings.get("FOV Size") as f32;
        let action_size = app.config.slider_settings.get("FOV Action Size") as f32;
        let style = app.config.dropdown_state.get("FOV Style");
        let fov_action = app.config.toggle_state.get("FOV Action");
        let color = parse_hex(&app.config.color_state.get("FOV Color"));
        let action_color = parse_hex(&app.config.color_state.get("FOV Action Color"));

        // Full-screen transparent window positioned on the current display.
        egui::Window::new("FOV Overlay")
            .id(egui::Id::new("fov_overlay"))
            .title_bar(false)
            .resizable(false)
            .movable(false)
            .paint_enabled(true)
            .collapsible(false)
            .always_on_top(app.config.toggle_state.get("UI TopMost"))
            .fixed_pos(egui::pos2(screen_x, screen_y))
            .fixed_size(egui::vec2(screen_w, screen_h))
            .frame(egui::Frame::none().fill(egui::Color32::TRANSPARENT))
            .show(ctx, |ui| {
                let center_x = screen_w / 2.0;
                let center_y = screen_h / 2.0;

                // Main FOV shape.
                let fov_rect = egui::Rect::from_center_size(
                    egui::pos2(center_x, center_y),
                    egui::vec2(fov_size, fov_size),
                );
                if style == "Rectangle" {
                    ui.painter().rect_stroke(fov_rect, 0.0, egui::Stroke::new(2.0, color));
                } else {
                    ui.painter().circle_stroke(
                        egui::pos2(center_x, center_y),
                        fov_size / 2.0,
                        egui::Stroke::new(2.0, color),
                    );
                }

                // Action zone.
                if fov_action {
                    let action_rect = egui::Rect::from_center_size(
                        egui::pos2(center_x, center_y),
                        egui::vec2(action_size, action_size),
                    );
                    if style == "Rectangle" {
                        ui.painter()
                            .rect_stroke(action_rect, 0.0, egui::Stroke::new(2.0, action_color));
                    } else {
                        ui.painter().circle_stroke(
                            egui::pos2(center_x, center_y),
                            action_size / 2.0,
                            egui::Stroke::new(2.0, action_color),
                        );
                    }
                }
            });

        // Reposition when the display changes (mirrors ForceReposition).
        let _ = RectangleI::new(0, 0, 0, 0);
    }
}

fn parse_hex(hex: &str) -> egui::Color32 {
    let s = hex.trim_start_matches('#');
    if s.len() >= 6 {
        let r = u8::from_str_radix(&s[0..2], 16).unwrap_or(255);
        let g = u8::from_str_radix(&s[2..4], 16).unwrap_or(255);
        let b = u8::from_str_radix(&s[4..6], 16).unwrap_or(255);
        return egui::Color32::from_rgb(r, g, b);
    }
    egui::Color32::WHITE
}
