use aimmy_core::config::AppConfig;
use eframe::egui;

/// Renders the detected-player ESP (box, confidence label, tracers) over the
/// current display. Mirrors the C# `DetectedPlayerWindow`.
pub struct EspOverlay;

impl EspOverlay {
    pub fn render(ctx: &egui::Context, app: &crate::AimmyApp) {
        let display = &app.display;
        let screen_w = display.screen_width() as f32;
        let screen_h = display.screen_height() as f32;
        let screen_x = display.screen_left() as f32;
        let screen_y = display.screen_top() as f32;

        let state = app.aim_state.lock().unwrap();
        let target = state.target.as_ref();
        let ai_conf = state.ai_conf;

        if target.is_none() && !app.config.toggle_state.get("Show Detected Player") {
            drop(state);
            return;
        }
        drop(state);

        egui::Window::new("ESP Overlay")
            .id(egui::Id::new("esp_overlay"))
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
                let color = parse_hex(&app.config.color_state.get("Detected Player Color"));
                let opacity = app.config.slider_settings.get("Opacity");

                if let Some(p) = app.aim_state.lock().unwrap().target.as_ref() {
                    let rect = p.rectangle;
                    let scale_x = screen_w / 640.0; // assume 640-base for overlay; adjust to image size
                    let scale_y = screen_h / 640.0;
                    let box_rect = egui::Rect::from_min_max(
                        egui::pos2(rect.x * scale_x, rect.y * scale_y),
                        egui::pos2((rect.x + rect.width) * scale_x, (rect.y + rect.height) * scale_y),
                    );

                    let stroke = egui::Stroke::new(
                        app.config.slider_settings.get("Border Thickness"),
                        color.linear_multiply(opacity),
                    );
                    ui.painter().rect_stroke(box_rect, 0.0, stroke);

                    if app.config.toggle_state.get("Show AI Confidence") {
                        let label = format!("{}: {:.1}%", p.class_name, p.confidence * 100.0);
                        let font_size = app.config.slider_settings.get("AI Confidence Font Size") as f32;
                        ui.painter().text(
                            egui::pos2(box_rect.center_x(), box_rect.min.y - font_size - 2.0),
                            egui::Align2::CENTER_CENTER,
                            label,
                            egui::FontId::proportional(font_size),
                            color.linear_multiply(opacity),
                        );
                    }
                }

                // Tracers from screen edge to target.
                if app.config.toggle_state.get("Show Tracers") {
                    if let Some(p) = app.aim_state.lock().unwrap().target.as_ref() {
                        let tracer_position = app.config.dropdown_state.get("Tracer Position");
                        let rect = p.rectangle;
                        let scale_x = screen_w / 640.0;
                        let scale_y = screen_h / 640.0;
                        let box_center_x = (rect.x + rect.width / 2.0) * scale_x;
                        let box_center_y = (rect.y + rect.height / 2.0) * scale_y;
                        let box_top = rect.y * scale_y;
                        let box_bottom = (rect.y + rect.height) * scale_y;
                        let box_left = rect.x * scale_x;
                        let box_right = (rect.x + rect.width) * scale_x;

                        let stroke = egui::Stroke::new(1.0, color.linear_multiply(opacity));
                        match tracer_position.as_str() {
                            "Top" => {
                                ui.painter().line_segment(
                                    [egui::pos2(box_center_x, 0.0), egui::pos2(box_center_x, box_top)],
                                    stroke,
                                );
                            }
                            "Middle" => {
                                let screen_h_center = screen_w / 2.0;
                                let (x, y) = if box_center_x < screen_h_center {
                                    (box_right, box_top + (box_bottom - box_top) / 2.0)
                                } else {
                                    (box_left, box_top + (box_bottom - box_top) / 2.0)
                                };
                                ui.painter().line_segment(
                                    [egui::pos2(screen_w / 2.0, screen_h / 2.0), egui::pos2(x, y)],
                                    stroke,
                                );
                            }
                            _ => {
                                ui.painter().line_segment(
                                    [
                                        egui::pos2(box_center_x, screen_h),
                                        egui::pos2(box_center_x, box_bottom),
                                    ],
                                    stroke,
                                );
                            }
                        }
                    }
                }
            });
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
