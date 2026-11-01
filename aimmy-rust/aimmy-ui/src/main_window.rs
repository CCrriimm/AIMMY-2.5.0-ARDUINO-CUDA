use aimmy_bindings::InputBindingManager;
use aimmy_core::config::AppConfig;
use aimmy_core::logging::LogManager;
use aimmy_inference::ExecutionProvider;
use aimmy_input::MouseMethod;
use eframe::egui;

use crate::{AimmyApp, FovOverlay, EspOverlay, PredictionMethod, widgets};

/// Render the Aim menu (aim assist, config, predictions, trigger bot, FOV, ESP).
pub fn aim_menu(ui: &mut egui::Ui, app: &mut AimmyApp) {
    ui.heading("Aim Assist");
    ui.separator();

    widgets::toggle(ui, &mut app.config, "Aim Assist", "Aim Assist");
    if app.config.toggle_state.get("Aim Assist") && app.selected_model.is_none() {
        ui.colored_label(egui::Color32::YELLOW, "Please load a model first.");
    }

    widgets::toggle(ui, &mut app.config, "Constant AI Tracking", "Constant AI Tracking");
    widgets::toggle(ui, &mut app.config, "Sticky Aim", "Sticky Aim");
    app.sticky_aim.set_enabled(app.config.toggle_state.get("Sticky Aim"));

    if app.config.toggle_state.get("Sticky Aim") {
        widgets::slider(
            ui,
            &mut app.config,
            "Sticky Aim Threshold",
            "Pixels",
            0.0..=100.0,
        );
    }

    ui.label("Keybinds:");
    widgets::KeyChanger::new().show(
        ui,
        &mut app.config,
        "Aim Keybind",
        "Aim Keybind",
        &mut app.bindings,
    );
    widgets::KeyChanger::new().show(
        ui,
        &mut app.config,
        "Second Aim Keybind",
        "Second Aim Keybind",
        &mut app.bindings,
    );
    widgets::KeyChanger::new().show(
        ui,
        &mut app.config,
        "Toggle Aim Keybind",
        "Toggle Aim Keybind",
        &mut app.bindings,
    );

    ui.separator();
    widgets::toggle(ui, &mut app.config, "Insta-Flick", "Insta-Flick");
    widgets::KeyChanger::new().show(
        ui,
        &mut app.config,
        "Insta-Flick Keybind",
        "Insta-Flick Keybind",
        &mut app.bindings,
    );
    widgets::KeyChanger::new().show(
        ui,
        &mut app.config,
        "Second Insta-Flick Keybind",
        "Second Insta-Flick Keybind",
        &mut app.bindings,
    );
    widgets::slider(
        ui,
        &mut app.config,
        "Insta-Flick Speed",
        "Speed",
        0.01..=1.0,
    );

    ui.separator();
    ui.heading("Aim Config");
    widgets::dropdown(
        ui,
        &mut app.config,
        "Mouse Movement Method",
        "Mouse Movement Method",
        &["Arduino", "ViGEmBus", "Makcu"],
    );
    let method = MouseMethod::from(app.config.dropdown_state.get("Mouse Movement Method").as_str());
    if method == MouseMethod::Arduino || method == MouseMethod::Makcu {
        widgets::dropdown(
            ui,
            &mut app.config,
            "Arduino COM Port",
            "Arduino COM Port",
            &[],
        );
        if method == MouseMethod::Makcu {
            widgets::dropdown(
                ui,
                &mut app.config,
                "Makcu Baud Rate",
                "Makcu Baud Rate",
                &["4000000", "115200"],
            );
        }
        if ui.button("Refresh COM Ports").clicked() {
            LogManager::info("COM ports refreshed.");
        }
    }

    widgets::dropdown(
        ui,
        &mut app.config,
        "Movement Path",
        "Movement Path",
        &["Cubic Bezier", "Linear", "Exponential", "Adaptive", "Perlin Noise"],
    );
    widgets::dropdown(
        ui,
        &mut app.config,
        "Detection Area Type",
        "Detection Area Type",
        &["Closest to Center Screen", "Closest to Mouse"],
    );
    widgets::dropdown(
        ui,
        &mut app.config,
        "Aiming Boundaries Alignment",
        "Aiming Boundaries Alignment",
        &["Center", "Top", "Bottom"],
    );

    widgets::slider(ui, &mut app.config, "Mouse Sensitivity (+/-)", "Sensitivity", 0.01..=1.0);
    widgets::toggle(
        ui,
        &mut app.config,
        "Adaptive Sensitivity Control",
        "Adaptive Sensitivity Control",
    );
    widgets::slider(ui, &mut app.config, "Overshoot Damping", "Damping", 0.05..=0.95);
    widgets::slider(ui, &mut app.config, "Overshoot Recovery", "Recovery", 0.05..=1.0);
    widgets::slider(ui, &mut app.config, "Mouse Jitter", "Jitter", 0.0..=15.0);

    widgets::toggle(ui, &mut app.config, "Y Axis Percentage Adjustment", "Y Axis %");
    widgets::toggle(ui, &mut app.config, "X Axis Percentage Adjustment", "X Axis %");
    widgets::slider(ui, &mut app.config, "Y Offset (Up/Down)", "Y Offset", -150.0..=150.0);
    widgets::slider(ui, &mut app.config, "Y Offset (%)", "Y Offset %", 0.0..=100.0);
    widgets::slider(ui, &mut app.config, "X Offset (Left/Right)", "X Offset", -150.0..=150.0);
    widgets::slider(ui, &mut app.config, "X Offset (%)", "X Offset %", 0.0..=100.0);

    ui.separator();
    ui.heading("Predictions");
    widgets::toggle(ui, &mut app.config, "Predictions", "Predictions");
    widgets::dropdown(
        ui,
        &mut app.config,
        "Prediction Method",
        "Prediction Method",
        &["Kalman Filter", "Shall0e's Prediction", "wisethef0x's EMA Prediction"],
    );
    let pm = PredictionMethod::from(app.config.dropdown_state.get("Prediction Method").as_str());
    match pm {
        PredictionMethod::KalmanFilter => {
            widgets::slider(ui, &mut app.config, "Kalman Lead Time", "Seconds", 0.02..=0.30);
        }
        PredictionMethod::WiseTheFoxEma => {
            widgets::slider(ui, &mut app.config, "WiseTheFox Lead Time", "Seconds", 0.02..=0.30);
        }
        PredictionMethod::Shalloe => {
            widgets::slider(ui, &mut app.config, "Shalloe Lead Multiplier", "Frames", 1.0..=10.0);
        }
    }
    widgets::toggle(ui, &mut app.config, "Flick Aim", "Flick Aim");
    widgets::slider(ui, &mut app.config, "Flick Aim Strength", "Amount", 0.01..=1.0);
    widgets::toggle(ui, &mut app.config, "Prediction Pixel Offset Enabled", "Prediction Pixel Offset");
    widgets::slider(ui, &mut app.config, "Prediction Pixel Offset", "Pixels", 0.0..=150.0);
    widgets::toggle(ui, &mut app.config, "EMA Smoothening", "EMA Smoothening");
    widgets::slider(ui, &mut app.config, "EMA Smoothening", "Amount", 0.01..=1.0);
    widgets::toggle(ui, &mut app.config, "Accelerated Smoothing", "Accelerated Smoothing");
    widgets::slider(ui, &mut app.config, "Accelerated Smoothing", "Amount", 0.01..=3.0);

    ui.separator();
    ui.heading("Auto Trigger");
    widgets::toggle(ui, &mut app.config, "Auto Trigger", "Auto Trigger");
    widgets::toggle(ui, &mut app.config, "Cursor Check", "Cursor Check");
    widgets::toggle(ui, &mut app.config, "Spray Mode", "Spray Mode");
    widgets::slider(ui, &mut app.config, "Auto Trigger Delay", "Seconds", 0.01..=1.0);

    ui.separator();
    ui.heading("FOV Config");
    widgets::toggle(ui, &mut app.config, "FOV", "FOV");
    app.show_fov = app.config.toggle_state.get("FOV");
    widgets::toggle(ui, &mut app.config, "FOV Action", "FOV Action");
    widgets::toggle(ui, &mut app.config, "Dynamic FOV", "Dynamic FOV");
    widgets::toggle(ui, &mut app.config, "Third Person Support", "Third Person Support");
    widgets::KeyChanger::new().show(
        ui,
        &mut app.config,
        "Dynamic FOV Keybind",
        "Dynamic FOV Keybind",
        &mut app.bindings,
    );
    widgets::dropdown(
        ui,
        &mut app.config,
        "FOV Style",
        "FOV Style",
        &["Circle", "Rectangle"],
    );
    widgets::color_changer(ui, &mut app.config, "FOV Color", "FOV Color");
    widgets::color_changer(ui, &mut app.config, "FOV Action Color", "FOV Action Color");
    widgets::slider(ui, &mut app.config, "FOV Size", "Size", 10.0..=640.0);
    widgets::slider(ui, &mut app.config, "FOV Action Size", "Size", 10.0..=640.0);
    widgets::slider(ui, &mut app.config, "Dynamic FOV Size", "Size", 10.0..=640.0);

    ui.separator();
    ui.heading("ESP Config");
    widgets::toggle(ui, &mut app.config, "Show Detected Player", "Show Detected Player");
    app.show_esp = app.config.toggle_state.get("Show Detected Player");
    widgets::toggle(ui, &mut app.config, "Show AI Confidence", "Show AI Confidence");
    widgets::toggle(ui, &mut app.config, "Show Tracers", "Show Tracers");
    widgets::dropdown(
        ui,
        &mut app.config,
        "Tracer Position",
        "Tracer Position",
        &["Bottom", "Middle", "Top"],
    );
    widgets::color_changer(ui, &mut app.config, "Detected Player Color", "Detected Player Color");
    widgets::slider(ui, &mut app.config, "AI Confidence Font Size", "Size", 1.0..=30.0);
    widgets::slider(ui, &mut app.config, "Corner Radius", "Radius", 0.0..=100.0);
    widgets::slider(ui, &mut app.config, "Border Thickness", "Thickness", 0.1..=10.0);
    widgets::slider(ui, &mut app.config, "Opacity", "Opacity", 0.0..=1.0);
}

/// Render the Model menu (model list, load, execution provider).
pub fn model_menu(ui: &mut egui::Ui, app: &mut AimmyApp) {
    ui.heading("Model");
    ui.separator();

    widgets::dropdown(
        ui,
        &mut app.config,
        "Execution Provider",
        "Execution Provider",
        &["CUDA", "TensorRT", "DirectML", "CPU"],
    );
    widgets::dropdown(
        ui,
        &mut app.config,
        "Image Size",
        "Image Size",
        &["640", "512", "416", "320", "256", "160"],
    );
    widgets::dropdown(
        ui,
        &mut app.config,
        "Screen Capture Method",
        "Screen Capture Method",
        &["DirectX", "GDI+"],
    );
    let capture_method = CaptureMethod::from(app.config.dropdown_state.get("Screen Capture Method").as_str());
    app.capture.set_capture_method(capture_method);

    ui.separator();
    ui.label("Available models:");
    egui::ScrollArea::vertical().show(ui, |ui| {
        for model in &app.models {
            let selected = app.selected_model.as_deref() == Some(model);
            if ui.selectable_label(selected, model).clicked() {
                app.selected_model = Some(model.clone());
                let path = app.models_dir.join(model);
                let provider = ExecutionProvider::from_str(&app.config.dropdown_state.get("Execution Provider"));
                let size: usize = app.config.dropdown_state.get("Image Size").parse().unwrap_or(640);
                app.session.load_model(&path, provider, size);
                LogManager::info(&format!("Loaded model: {model}"));
            }
        }
        if app.models.is_empty() {
            ui.label("No .onnx models found in bin/models");
        }
    });

    if let Some(m) = &app.selected_model {
        ui.separator();
        ui.label(format!("Loaded: {m}"));
        if ui.button("Reload model").clicked() {
            let path = app.models_dir.join(m);
            let provider = ExecutionProvider::from_str(&app.config.dropdown_state.get("Execution Provider"));
            let size: usize = app.config.dropdown_state.get("Image Size").parse().unwrap_or(640);
            app.session.load_model(&path, provider, size);
        }
    }
}

/// Render the Settings menu.
pub fn settings_menu(ui: &mut egui::Ui, app: &mut AimmyApp) {
    ui.heading("Settings");
    ui.separator();

    widgets::toggle(ui, &mut app.config, "Realtime GPU Priority", "Realtime GPU Priority");
    widgets::toggle(ui, &mut app.config, "Collect Data While Playing", "Collect Data While Playing");
    widgets::toggle(ui, &mut app.config, "Auto Label Data", "Auto Label Data");
    widgets::toggle(ui, &mut app.config, "Mouse Background Effect", "Mouse Background Effect");
    widgets::toggle(ui, &mut app.config, "UI TopMost", "UI TopMost");
    widgets::toggle(ui, &mut app.config, "Debug Mode", "Debug Mode");
    widgets::toggle(ui, &mut app.config, "StreamGuard", "StreamGuard");

    ui.separator();
    widgets::color_changer(ui, &mut app.config, "Theme Color", "Theme Color");

    ui.separator();
    if ui.button("Save configuration").clicked() {
        app.save_config();
    }
    if ui.button("Load configuration").clicked() {
        app.load_config();
    }
}

/// Render the About menu.
pub fn about_menu(ui: &mut egui::Ui, _app: &mut AimmyApp) {
    ui.heading("About");
    ui.separator();
    ui.label("Aimmy 2.5.0 — Rust rewrite");
    ui.label("Universal AI-Based Aim Alignment Mechanism.");
    ui.label("Features: config, FOV, ESP, sticky aim, predictions, auto-trigger, Arduino/Makcu/ViGEm.");
    ui.hyperlink("https://aimmy.dev/");
}

// -----------------------------------------------------------------------------

use aimmy_capture::CaptureMethod;
