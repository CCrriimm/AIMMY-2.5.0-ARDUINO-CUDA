use aimmy_bindings::InputBindingManager;
use aimmy_capture::{CaptureManager, CaptureMethod, DisplayManager};
use aimmy_core::config::AppConfig;
use aimmy_core::logging::{LogManager, LogLevel};
use aimmy_core::types::{Prediction as CorePrediction, RectangleF};
use aimmy_inference::{
    calculate_coordinates, detection_box, fov_bounds, parse_output, build_input_tensor,
    sticky_aim::StickyAim, ExecutionProvider, ModelFormat, OnnxSession,
};
use aimmy_inference::model::{auto_trigger, TriggerAction};
use aimmy_input::{MouseManager, MouseMethod};
use eframe::egui;
use ndarray::{Array, Ix4};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub mod main_window;
pub mod fov_overlay;
pub mod esp_overlay;
pub mod widgets;

pub use main_window::MainWindow;
pub use fov_overlay::FovOverlay;
pub use esp_overlay::EspOverlay;

/// Shared aim state written by the AI loop and read by the UI overlays.
#[derive(Default)]
pub struct AimState {
    pub target: Option<CorePrediction>,
    pub ai_conf: f32,
    pub processing: bool,
}

/// Top-level application.
pub struct AimmyApp {
    pub config: AppConfig,
    pub display: DisplayManager,
    pub capture: CaptureManager,
    pub mouse: MouseManager,
    pub bindings: InputBindingManager,
    pub session: OnnxSession,
    pub sticky_aim: StickyAim,
    pub kalman: aimmy_inference::predictors::kalman::KalmanFilter,
    pub ema: aimmy_inference::predictors::ema::EmaPredictor,
    pub shalloe: aimmy_inference::predictors::shalloe::ShalloePredictor,

    pub aim_state: Arc<Mutex<AimState>>,

    last_detection_box: Option<RectangleF>,
    last_ai_conf: f32,

    aim_held: bool,
    second_aim_held: bool,
    insta_held: bool,
    second_insta_held: bool,
    dynamic_fov_held: bool,
    emergency_stop: bool,
    model_switch: bool,

    current_menu: String,
    show_fov: bool,
    show_esp: bool,

    models_dir: PathBuf,
    models: Vec<String>,
    selected_model: Option<String>,
}

impl AimmyApp {
    pub fn new() -> Self {
        LogManager::init("bin", "aimmy.log");
        LogManager::info("Aimmy Rust starting up.");

        let config = AppConfig::default();

        let mut display = DisplayManager::new();
        display.initialize();

        let screen_w = display.screen_width() as f64;
        let screen_h = display.screen_height() as f64;

        let mut capture = CaptureManager::new();
        let method = CaptureMethod::from(config.dropdown_state.get("Screen Capture Method").as_str());
        capture.set_capture_method(method);

        let mut mouse = MouseManager::new();
        mouse.set_screen_size(screen_w, screen_h);
        mouse.set_method(MouseMethod::from(config.dropdown_state.get("Mouse Movement Method").as_str()));

        let port = config.dropdown_state.get("Arduino COM Port");
        if !port.is_empty() {
            let method = config.dropdown_state.get("Mouse Movement Method");
            if method == "Makcu" {
                let baud: u32 = config.dropdown_state.get("Makcu Baud Rate").parse().unwrap_or(4_000_000);
                mouse.init_makcu(&port, baud);
            } else {
                mouse.init_arduino(&port);
            }
        }

        let bindings = InputBindingManager::new();
        for (id, key) in InputBindingManager::default_bindings() {
            let saved = config.binding_settings.get(&id);
            let key = if saved.is_empty() { key } else { saved };
            bindings.setup_default(&id, &key);
        }

        let session = OnnxSession::new();
        let mut sticky_aim = StickyAim::new();
        sticky_aim.set_enabled(config.toggle_state.get("Sticky Aim"));

        let models_dir = PathBuf::from("bin/models");
        let models = list_models(&models_dir);

        Self {
            config,
            display,
            capture,
            mouse,
            bindings,
            session,
            sticky_aim,
            kalman: aimmy_inference::predictors::kalman::KalmanFilter::new(),
            ema: aimmy_inference::predictors::ema::EmaPredictor::new(),
            shalloe: aimmy_inference::predictors::shalloe::ShalloePredictor::new(),
            aim_state: Arc::new(Mutex::new(AimState::default())),
            last_detection_box: None,
            last_ai_conf: 0.0,
            aim_held: false,
            second_aim_held: false,
            insta_held: false,
            second_insta_held: false,
            dynamic_fov_held: false,
            emergency_stop: false,
            model_switch: false,
            current_menu: "AimMenu".into(),
            show_fov: config.toggle_state.get("FOV"),
            show_esp: config.toggle_state.get("Show Detected Player"),
            models_dir,
            models,
            selected_model: None,
        }
    }

    pub fn start_ai_loop(&mut self) {
        LogManager::info("AI loop started.");
    }

    pub fn stop_ai_loop(&mut self) {
        LogManager::info("AI loop stopped.");
    }

    /// One frame of the aim loop. Called from `eframe::update` whenever any
    // feature requires processing.
    pub fn aim_tick(&mut self) {
        if !should_process(&self.config) {
            self.mouse.reset_movement();
            return;
        }

        let screen_w = self.display.screen_width() as f32;
        let screen_h = self.display.screen_height() as f32;
        let (target_x, target_y) = (screen_w / 2.0, screen_h / 2.0);

        let image_size = self.session.image_size() as i32;
        let box_ = detection_box(target_x as i32, target_y as i32, image_size);

        let frame = self.capture.screen_grab(&box_);
        let Some(frame) = frame else {
            self.mouse.reset_movement();
            return;
        };

        // Preprocess + run inference (stubbed when onnx feature off).
        let _input = build_input_tensor(&frame.pixels, image_size as usize);
        let detections = self.run_inference(image_size as usize);

        let fov_size = self.config.slider_settings.get("FOV Size") as f32;
        let (fov_min_x, fov_max_x, fov_min_y, fov_max_y) = fov_bounds(image_size as f32, fov_size);

        let min_conf = self.config.slider_settings.get("AI Minimum Confidence") / 100.0;
        let target_class = self.config.dropdown_state.get("Target Class");
        let selected_class = if target_class == "Best Confidence" {
            None
        } else {
            self.session.classes().iter().find(|(_, v)| *v == target_class).map(|(k, _)| *k)
        };

        let preds = parse_output(
            &detections,
            [1, image_size as usize / 8, image_size as usize / 8], // placeholder dims
            ModelFormat::YoloRawChannelsFirst,
            image_size as f32,
            fov_min_x,
            fov_max_x,
            min_conf as f32,
            self.session.classes(),
            selected_class,
            box_.left as f32,
            box_.top as f32,
        );

        let center = image_size as f32 / 2.0;
        let mut best: Option<&CorePrediction> = None;
        let mut best_dist = f32::INFINITY;
        for p in &preds {
            let cx = p.center_x_translated * image_size as f32;
            let cy = p.center_y_translated * image_size as f32;
            let d2 = (cx - center).powi(2) + (cy - center).powi(2);
            if d2 < best_dist {
                best_dist = d2;
                best = Some(p);
            }
        }

        let action_size = self.config.slider_settings.get("FOV Action Size") as f32;
        let style = self.config.dropdown_state.get("FOV Style");
        let sticky = self.sticky_aim.resolve(
            best,
            &preds,
            image_size as f32,
            self.config.toggle_state.get("FOV Action"),
            action_size / 2.0,
            &style,
            self.config.slider_settings.get("Sticky Aim Threshold"),
            fov_size,
        );

        let scale_x = screen_w / image_size as f32;
        let scale_y = screen_h / image_size as f32;
        let alignment = self.config.dropdown_state.get("Aiming Boundaries Alignment");

        let (detected_x, detected_y) = if let Some(p) = &sticky {
            calculate_coordinates(p, scale_x, scale_y, screen_w, screen_h, &self.config, &alignment)
        } else {
            (0, 0)
        };

        // Expose target for the ESP overlay.
        if let Some(p) = &sticky {
            self.last_ai_conf = p.confidence;
        }
        {
            let mut state = self.aim_state.lock().unwrap();
            state.target = sticky.clone();
            state.ai_conf = self.last_ai_conf;
            state.processing = true;
        }
        self.last_detection_box = sticky.as_ref().map(|p| p.rectangle);

        // Auto trigger.
        let cursor_in_box = false;
        let action = auto_trigger(
            &self.config,
            self.aim_held,
            self.second_aim_held,
            best,
            self.last_detection_box,
            cursor_in_box,
        );
        match action {
            TriggerAction::Click | TriggerAction::Hold => {
                self.mouse.do_trigger_click(&self.config, self.aim_held, self.second_aim_held, None, false);
            }
            TriggerAction::Release => {
                self.mouse.reset_spray_state();
            }
            TriggerAction::None => {}
        }

        // Aim movement.
        if self.config.toggle_state.get("Aim Assist")
            && (self.config.toggle_state.get("Constant AI Tracking")
                || self.aim_held
                || self.second_aim_held)
        {
            if let Some(p) = &sticky {
                let (dx, dy) = (detected_x, detected_y);
                // Apply selected prediction method to produce aim point.
                let (px, py) = self.predict_point(p, dx as f32, dy as f32);
                self.mouse.move_crosshair(&self.config, px as i32, py as i32);
            } else {
                self.mouse.reset_movement();
            }
        } else {
            self.mouse.reset_movement();
        }
    }

    /// Apply the selected prediction method to the raw aim point.
    fn predict_point(&mut self, p: &CorePrediction, x: f32, y: f32) -> (f32, f32) {
        let method = crate::PredictionMethod::from(self.config.dropdown_state.get("Prediction Method").as_str());
        match method {
            crate::PredictionMethod::KalmanFilter => {
                let lead = self.config.slider_settings.get("Kalman Lead Time") as f64;
                self.kalman.update(x as f64, y as f64);
                let (px, py) = self.kalman.predict(lead);
                (px as f32, py as f32)
            }
            crate::PredictionMethod::WiseTheFoxEma => {
                let lead = self.config.slider_settings.get("WiseTheFox Lead Time") as f64;
                self.ema.update(x as f64, y as f64);
                let (px, py) = self.ema.estimate(lead);
                (px as f32, py as f32)
            }
            crate::PredictionMethod::Shalloe => {
                let mult = self.config.slider_settings.get("Shalloe Lead Multiplier") as f64;
                self.shalloe.update(x as f64, y as f64);
                let (px, py) = self.shalloe.predict(mult);
                (px as f32, py as f32)
            }
        }
    }

    fn run_inference(&mut self, _size: usize) -> Vec<f32> {
        // Real ONNX execution goes here behind the `onnx` feature flag.
        vec![]
    }

    pub fn handle_keybind_pressed(&mut self, id: &str) {
        match id {
            "Aim Keybind" => self.aim_held = true,
            "Second Aim Keybind" => self.second_aim_held = true,
            "Insta-Flick Keybind" => self.insta_held = true,
            "Second Insta-Flick Keybind" => self.second_insta_held = true,
            "Dynamic FOV Keybind" => self.dynamic_fov_held = true,
            "Emergency Stop Keybind" => {
                self.config.toggle_state.set("Aim Assist", false);
                self.config.toggle_state.set("Constant AI Tracking", false);
                self.config.toggle_state.set("Auto Trigger", false);
                LogManager::info("[Emergency Stop] Disabled all AI features.");
            }
            "Model Switch Keybind" => self.cycle_model(),
            "Toggle Aim Keybind" => {
                let new_state = !self.config.toggle_state.get("Aim Assist");
                self.config.toggle_state.set("Aim Assist", new_state);
            }
            _ => {}
        }
    }

    pub fn handle_keybind_released(&mut self, id: &str) {
        if id == "Dynamic FOV Keybind" {
            self.dynamic_fov_held = false;
        }
    }

    /// Handle an insta-flick request (called when flick keybind is held).
    pub fn do_insta_flick(&mut self, detected_x: i32, detected_y: i32) {
        self.mouse.insta_flick(&self.config, detected_x, detected_y);
    }

    fn cycle_model(&mut self) {
        if self.models.is_empty() {
            return;
        }
        let idx = self
            .selected_model
            .as_ref()
            .and_then(|m| self.models.iter().position(|x| x == m))
            .unwrap_or(0);
        let next = (idx + 1) % self.models.len();
        self.selected_model = Some(self.models[next].clone());
        if let Some(m) = &self.selected_model {
            let path = self.models_dir.join(m);
            let provider = ExecutionProvider::from_str(&self.config.dropdown_state.get("Execution Provider"));
            let size: usize = self.config.dropdown_state.get("Image Size").parse().unwrap_or(640);
            self.session.load_model(&path, provider, size);
        }
    }

    pub fn save_config(&self) {
        let _ = std::fs::create_dir_all("bin/configs");
        let _ = self.config.save("bin/configs/Default.cfg");
        LogManager::info("Configuration saved.");
    }

    pub fn load_config(&mut self) {
        match AppConfig::load("bin/configs/Default.cfg") {
            Ok(cfg) => self.config = cfg,
            Err(e) => LogManager::warn(&format!("Failed to load config: {e}")),
        }
    }
}

impl eframe::App for AimmyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Run the aim loop whenever a feature needs processing.
        if should_process(&self.config)
            || self.config.toggle_state.get("Show Detected Player")
            || self.config.toggle_state.get("Insta-Flick")
        {
            self.aim_tick();
        } else {
            self.mouse.reset_movement();
        }

        // Top menu bar.
        egui::TopBottomPanel::top("top_menu").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Aimmy 2.5.0 (Rust)");
                ui.separator();
                for (label, name) in [
                    ("Aim", "AimMenu"),
                    ("Model", "ModelMenu"),
                    ("Settings", "SettingsMenu"),
                    ("About", "AboutMenu"),
                ] {
                    if ui.selectable_label(self.current_menu == name, label).clicked() {
                        self.current_menu = name.into();
                    }
                }
                ui.separator();
                ui.label(format!("Provider: {:?}", self.session.provider()));
                ui.label(format!("Image size: {}", self.session.image_size()));
                ui.separator();
                if ui.button("Save Config").clicked() {
                    self.save_config();
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            match self.current_menu.as_str() {
                "AimMenu" => MainWindow::aim_menu(ui, self),
                "ModelMenu" => MainWindow::model_menu(ui, self),
                "SettingsMenu" => MainWindow::settings_menu(ui, self),
                "AboutMenu" => MainWindow::about_menu(ui, self),
                _ => MainWindow::aim_menu(ui, self),
            }
        });

        // Transparent overlays drawn with egui (full-screen windows would be
        // separate winit viewports in a production build).
        if self.show_fov {
            FovOverlay::render(ctx, self);
        }
        if self.show_esp {
            EspOverlay::render(ctx, self);
        }

        ctx.request_repaint_after(Duration::from_millis(1));
    }

    fn on_exit(&mut self, _gl: Option<&eframe::gl::Gl>) {
        self.stop_ai_loop();
        self.mouse.reset_spray_state();
        let _ = self.save_config();
        LogManager::info("Aimmy Rust shutting down.");
    }
}

fn should_process(cfg: &AppConfig) -> bool {
    cfg.toggle_state.get("Aim Assist")
        || cfg.toggle_state.get("Show Detected Player")
        || cfg.toggle_state.get("Auto Trigger")
        || cfg.toggle_state.get("Insta-Flick")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionMethod {
    KalmanFilter,
    Shalloe,
    WiseTheFoxEma,
}

impl Default for PredictionMethod {
    fn default() -> Self {
        PredictionMethod::KalmanFilter
    }
}

impl From<&str> for PredictionMethod {
    fn from(s: &str) -> Self {
        match s {
            "Shall0e's Prediction" => PredictionMethod::Shalloe,
            "wisethef0x's EMA Prediction" => PredictionMethod::WiseTheFoxEma,
            _ => PredictionMethod::KalmanFilter,
        }
    }
}

fn list_models(dir: &PathBuf) -> Vec<String> {
    std::fs::read_dir(dir)
        .ok()
        .and_then(|rd| {
            let mut v: Vec<String> = rd
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().map(|ext| ext == "onnx").unwrap_or(false))
                .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
                .collect();
            v.sort();
            Some(v)
        })
        .unwrap_or_default()
}
