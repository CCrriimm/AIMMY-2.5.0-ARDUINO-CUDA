use aimmy_core::config::AppConfig;
use aimmy_core::logging::LogManager;
use crate::loop_controller::AimLoopController;
use crate::movement_paths::{MovementPath, movement_path};
use crate::vigem::ViGEmInput;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCK};

/// Which physical/virtual mouse backend is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MouseMethod {
    #[default]
    Arduino,
    ViGEmBus,
    Makcu,
}

impl From<&str> for MouseMethod {
    fn from(s: &str) -> Self {
        match s {
            "ViGEmBus" => MouseMethod::ViGEmBus,
            "Makcu" => MouseMethod::Makcu,
            _ => MouseMethod::Arduino,
        }
    }
}

/// Result of a movement calculation: the delta to send to the mouse.
pub struct MoveDelta {
    pub x: i32,
    pub y: i32,
}

/// The mouse manager: filtering, deadband, movement paths, stability
/// controller, and dispatch to the active backend.
///
/// Mirrors the C# `MouseManager` + `AimLoopController` + `MovementPaths`.
pub struct MouseManager {
    screen_width: f64,
    screen_height: f64,
    method: MouseMethod,
    arduino: Option<Box<dyn ArduinoBackend>>,
    makcu: Option<Box<dyn MakcuBackend>>,
    vigem: ViGEmInput,

    last_click: Option<u64>,
    is_spraying: bool,

    previous_x: f64,
    previous_y: f64,
    sub_pixel_x: f64,
    sub_pixel_y: f64,
    filtered_target_x: f64,
    filtered_target_y: f64,
    last_target_x: f64,
    last_target_y: f64,
    oscillation_counter: i32,
    has_target: bool,

    pub smoothing_factor: f64,
    pub ema_smoothing_enabled: bool,

    loop_controller: AimLoopController,

    rng: fastrand::Rng,
}

trait ArduinoBackend {
    fn send_mouse_command(&mut self, x: i32, y: i32, click: i32);
    fn close(&mut self);
}

trait MakcuBackend {
    fn send_click(&mut self, down: i32);
    fn move_(&mut self, x: i32, y: i32);
    fn close(&mut self);
}

impl Default for MouseManager {
    fn default() -> Self {
        Self::new()
    }
}

impl MouseManager {
    pub fn new() -> Self {
        Self {
            screen_width: 1920.0,
            screen_height: 1080.0,
            method: MouseMethod::Arduino,
            arduino: None,
            makcu: None,
            vigem: ViGEmInput::new(),
            last_click: None,
            is_spraying: false,
            previous_x: 0.0,
            previous_y: 0.0,
            sub_pixel_x: 0.0,
            sub_pixel_y: 0.0,
            filtered_target_x: 0.0,
            filtered_target_y: 0.0,
            last_target_x: 0.0,
            last_target_y: 0.0,
            oscillation_counter: 0,
            has_target: false,
            smoothing_factor: 0.5,
            ema_smoothing_enabled: false,
            loop_controller: AimLoopController::new(),
            rng: fastrand::Rng::new(),
        }
    }

    pub fn set_screen_size(&mut self, w: f64, h: f64) {
        self.screen_width = w;
        self.screen_height = h;
    }

    pub fn set_method(&mut self, method: MouseMethod) {
        self.method = method;
    }

    pub fn method(&self) -> MouseMethod {
        self.method
    }

    /// Initialise Arduino on a COM port.
    pub fn init_arduino(&mut self, com_port: &str) {
        self.disconnect_arduino();
        match crate::arduino::ArduinoInput::open(com_port) {
            Ok(port) => {
                self.arduino = Some(Box::new(port));
                LogManager::info(&format!("Connected to Arduino on {com_port}"));
            }
            Err(e) => {
                LogManager::error(&format!("Failed to connect to Arduino on {com_port}: {e}"));
            }
        }
    }

    pub fn disconnect_arduino(&mut self) {
        if let Some(p) = self.arduino.as_mut() {
            let _ = p.close();
        }
        self.arduino = None;
    }

    /// Initialise Makcu on a COM port.
    pub fn init_makcu(&mut self, com_port: &str, baud_rate: u32) {
        self.disconnect_makcu();
        self.disconnect_arduino();
        match crate::makcu::MakcuInput::open(com_port, baud_rate) {
            Ok(port) => {
                self.makcu = Some(Box::new(port));
                LogManager::info(&format!("Connected to Makcu on {com_port} @ {baud_rate}"));
            }
            Err(e) => {
                LogManager::error(&format!("Failed to connect to Makcu on {com_port}: {e}"));
            }
        }
    }

    pub fn disconnect_makcu(&mut self) {
        if let Some(p) = self.makcu.as_mut() {
            let _ = p.close();
        }
        self.makcu = None;
    }

    /// Reset per-target filter state (called when aim is idle).
    pub fn reset_movement(&mut self) {
        self.previous_x = 0.0;
        self.previous_y = 0.0;
        self.sub_pixel_x = 0.0;
        self.sub_pixel_y = 0.0;
        self.filtered_target_x = 0.0;
        self.filtered_target_y = 0.0;
        self.last_target_x = 0.0;
        self.last_target_y = 0.0;
        self.oscillation_counter = 0;
        self.has_target = false;
        self.loop_controller.reset();
        if matches!(self.method, MouseMethod::ViGEmBus) {
            self.vigem.reset_movement();
        }
    }

    pub fn reset_spray_state(&mut self) {
        if self.is_spraying {
            self.release_mouse_button();
        }
        self.reset_movement();
    }

    /// Main entry: move the crosshair toward `detected_x`/`detected_y`
    /// (absolute screen coords), applying the full Aimmy movement pipeline.
    pub fn move_crosshair(&mut self, cfg: &AppConfig, detected_x: i32, detected_y: i32) {
        let half_w = self.screen_width / 2.0;
        let half_h = self.screen_height / 2.0;
        let raw_target_x = detected_x as f64 - half_w;
        let raw_target_y = detected_y as f64 - half_h;

        // ── Stage 1: zero-phase-lag deadband filter ──
        if !self.has_target {
            self.filtered_target_x = raw_target_x;
            self.filtered_target_y = raw_target_y;
            self.last_target_x = raw_target_x;
            self.last_target_y = raw_target_y;
            self.oscillation_counter = 0;
            self.has_target = true;
        } else {
            // Rule A: zero-crossing purge.
            if (raw_target_x > 0.0 && self.filtered_target_x < 0.0)
                || (raw_target_x < 0.0 && self.filtered_target_x > 0.0)
            {
                self.filtered_target_x = raw_target_x;
            }
            if (raw_target_y > 0.0 && self.filtered_target_y < 0.0)
                || (raw_target_y < 0.0 && self.filtered_target_y > 0.0)
            {
                self.filtered_target_y = raw_target_y;
            }

            // Rule B: deadband noise rejection.
            let dx = raw_target_x - self.filtered_target_x;
            let dy = raw_target_y - self.filtered_target_y;
            let jitter_dist = (dx * dx + dy * dy).sqrt();
            if jitter_dist < 2.0 {
                self.filtered_target_x = self.filtered_target_x * 0.95 + raw_target_x * 0.05;
                self.filtered_target_y = self.filtered_target_y * 0.95 + raw_target_y * 0.05;
            } else if jitter_dist < 6.0 {
                let t = (jitter_dist - 2.0) / 4.0;
                let alpha = 0.50 + t * 0.35;
                self.filtered_target_x = self.filtered_target_x * (1.0 - alpha) + raw_target_x * alpha;
                self.filtered_target_y = self.filtered_target_y * (1.0 - alpha) + raw_target_y * alpha;
            } else {
                self.filtered_target_x = self.filtered_target_x * 0.10 + raw_target_x * 0.90;
                self.filtered_target_y = self.filtered_target_y * 0.10 + raw_target_y * 0.90;
            }
        }

        let target_x = self.filtered_target_x;
        let target_y = self.filtered_target_y;
        let dist = (target_x * target_x + target_y * target_y).sqrt();

        // ── Stage 1b: closed-loop stability controller ──
        self.loop_controller.configure(
            cfg.toggle_state.get("Adaptive Sensitivity Control"),
            cfg.slider_settings.get("Overshoot Damping"),
            cfg.slider_settings.get("Overshoot Recovery"),
        );
        self.loop_controller.evaluate(target_x, target_y);
        let dt_scale = self.loop_controller.step_scale();

        // ── Stage 2: lock-on deadzone & smoothstep dampening ──
        let deadzone = self.loop_controller.lock_tolerance_pixels();
        const DAMPENING_RADIUS: f64 = 18.0;
        let mut lock_damping = 1.0;
        if dist <= deadzone {
            self.sub_pixel_x = 0.0;
            self.sub_pixel_y = 0.0;
            self.previous_x = 0.0;
            self.previous_y = 0.0;
            self.oscillation_counter = 0;
            self.last_target_x = target_x;
            self.last_target_y = target_y;
            self.loop_controller.record_command(target_x, target_y, 0.0, 0.0);
            return;
        }
        if dist < DAMPENING_RADIUS {
            let normalized = (dist - deadzone) / (DAMPENING_RADIUS - deadzone);
            lock_damping = normalized * normalized * (3.0 - 2.0 * normalized);
        }

        let aspect_ratio_correction = self.screen_width / self.screen_height;

        // Mouse jitter (humanisation).
        let mouse_jitter = cfg.slider_settings.get("Mouse Jitter") as f64;
        let jitter_scale = ((dist - 20.0) / 30.0).clamp(0.0, 1.0);
        let jitter_x = self.rng.f64() * mouse_jitter * 2.0 - mouse_jitter;
        let jitter_y = self.rng.f64() * mouse_jitter * 2.0 - mouse_jitter;

        // ── Stage 3: movement path ──
        let sens_setting = cfg.slider_settings.get("Mouse Sensitivity (+/-)");
        let sens_factor = self.loop_controller.effective_loop_fraction(sens_setting);

        let path: MovementPath = cfg.dropdown_state.get("Movement Path").as_str().into();
        let new_position = movement_path(
            path,
            (0.0, 0.0),
            (target_x, target_y),
            sens_factor,
            sens_setting,
        );

        let mut pos_x = new_position.0;
        let mut pos_y = new_position.1;

        if self.ema_smoothing_enabled {
            pos_x = self.previous_x * (1.0 - self.smoothing_factor) + pos_x * self.smoothing_factor;
            pos_y = self.previous_y * (1.0 - self.smoothing_factor) + pos_y * self.smoothing_factor;
        }

        // ── Stage 4: accelerated smoothing ──
        if cfg.toggle_state.get("Accelerated Smoothing") {
            let accel_amount = cfg.slider_settings.get("Accelerated Smoothing");
            let accel_factor = if dist < 25.0 {
                1.0 + accel_amount * 0.2 * (dist / 25.0)
            } else {
                (1.0 + (dist / 100.0) * accel_amount).clamp(1.0, 2.5)
            };
            pos_x *= accel_factor;
            pos_y *= accel_factor;
        }

        // ── Stage 5: flick aim ──
        if cfg.toggle_state.get("Flick Aim") && dist > 20.0 {
            let flick_strength = cfg.slider_settings.get("Flick Aim Strength");
            let range_weight = ((dist - 20.0) / 40.0).clamp(0.0, 1.0);
            let effective_flick = (flick_strength * 0.40 * range_weight * lock_damping).clamp(0.0, 0.40);
            pos_x = pos_x * (1.0 - effective_flick) + target_x * effective_flick;
            pos_y = pos_y * (1.0 - effective_flick) + target_y * effective_flick;
        }

        // ── Stage 6: derivative active braking ──
        let closing_dist = (self.last_target_x * self.last_target_x + self.last_target_y * self.last_target_y).sqrt() - dist;
        if closing_dist > 0.0 && dist < 30.0 {
            let brake_factor = (1.0 - (closing_dist / dist.max(4.0)) * 0.4).clamp(0.35, 1.0);
            pos_x *= brake_factor;
            pos_y *= brake_factor;
        }

        pos_x *= lock_damping;
        pos_y *= lock_damping;
        pos_y /= aspect_ratio_correction;
        pos_x += jitter_x;
        pos_y += jitter_y;

        // ── Stage 7: anti-hunting oscillation suppression ──
        if self.previous_x != 0.0 || self.previous_y != 0.0 {
            let dot = pos_x * self.previous_x + pos_y * self.previous_y;
            if dot < 0.0 {
                self.oscillation_counter = (self.oscillation_counter + 1).min(5);
            } else {
                self.oscillation_counter = (self.oscillation_counter - 1).max(0);
            }
            if self.oscillation_counter > 0 {
                let damp = (1.0 - (self.oscillation_counter as f64 * 0.30)).max(0.15);
                pos_x *= damp;
                pos_y *= damp;
            }
        }

        pos_x = pos_x.clamp(-150.0, 150.0);
        pos_y = pos_y.clamp(-150.0, 150.0);

        // ── Stage 8: frame-rate normalisation ──
        pos_x *= dt_scale;
        pos_y *= dt_scale;

        // ── Stage 9: legacy cap when adaptive controller off ──
        if !self.loop_controller.is_adaptive_enabled() && dist > 0.0 {
            const MAX_STEP_RATIO: f64 = 0.85;
            const MIN_STEP_CAP_PX: f64 = 6.0;
            let max_step = (MIN_STEP_CAP_PX).max(dist * MAX_STEP_RATIO);
            if pos_x.abs() > max_step {
                pos_x = pos_x.signum() * max_step;
            }
            if pos_y.abs() > max_step {
                pos_y = pos_y.signum() * max_step;
            }
        }

        // Sub-pixel accumulation.
        let total_x = pos_x + self.sub_pixel_x;
        let total_y = pos_y + self.sub_pixel_y;
        let move_x = total_x.round() as i32;
        let move_y = total_y.round() as i32;
        self.sub_pixel_x = total_x - move_x as f64;
        self.sub_pixel_y = total_y - move_y as f64;

        self.loop_controller.record_command(target_x, target_y, move_x as f64, move_y as f64);

        if move_x != 0 || move_y != 0 {
            match self.method {
                MouseMethod::ViGEmBus => self.vigem.move_(move_x, move_y),
                MouseMethod::Makcu => {
                    if let Some(p) = self.makcu.as_mut() {
                        p.move_(move_x, move_y);
                    }
                }
                MouseMethod::Arduino => {
                    if let Some(p) = self.arduino.as_mut() {
                        p.send_mouse_command(move_x, move_y, 0);
                    }
                }
            }
        }

        self.previous_x = pos_x;
        self.previous_y = pos_y;
        self.last_target_x = target_x;
        self.last_target_y = target_y;

        if !cfg.toggle_state.get("Auto Trigger") {
            self.reset_spray_state();
        }
    }

    /// Instant snap toward `detected_x`/`detected_y` (port of `InstaFlick`).
    pub fn insta_flick(&mut self, cfg: &AppConfig, detected_x: i32, detected_y: i32) {
        let half_w = self.screen_width / 2.0;
        let half_h = self.screen_height / 2.0;
        let mut target_x = detected_x as f64 - half_w;
        let mut target_y = detected_y as f64 - half_h;

        if target_x.abs() <= 1.0 && target_y.abs() <= 1.0 {
            return;
        }

        let aspect_ratio_correction = self.screen_width / self.screen_height;
        target_y /= aspect_ratio_correction;

        let flick_speed = cfg.slider_settings.get("Insta-Flick Speed");
        let step_multiplier = (1.0 / flick_speed.max(0.05) * 0.25).clamp(0.25, 1.0);

        let mut move_x = (target_x * step_multiplier) as i32;
        let mut move_y = (target_y * step_multiplier) as i32;

        const MAX_STEP_RATIO: f64 = 0.85;
        let flick_cap = (8.0f64).max(target_x.abs().max(target_y.abs()) * MAX_STEP_RATIO) as i32;
        move_x = move_x.clamp(-flick_cap, flick_cap);
        move_y = move_y.clamp(-flick_cap, flick_cap);
        move_x = move_x.clamp(-127, 127);
        move_y = move_y.clamp(-127, 127);

        if move_x != 0 || move_y != 0 {
            match self.method {
                MouseMethod::ViGEmBus => self.vigem.move_(move_x, move_y),
                MouseMethod::Makcu => {
                    if let Some(p) = self.makcu.as_mut() {
                        p.move_(move_x, move_y);
                    }
                }
                MouseMethod::Arduino => {
                    if let Some(p) = self.arduino.as_mut() {
                        p.send_mouse_command(move_x, move_y, 0);
                    }
                }
            }
        }

        self.previous_x = move_x as f64;
        self.previous_y = move_y as f64;
    }

    /// Perform a trigger click (single or spray hold/release).
    pub fn do_trigger_click(
        &mut self,
        cfg: &AppConfig,
        aim_held: bool,
        second_aim_held: bool,
        detection_box: Option<(f32, f32, f32, f32)>,
        cursor_in_box: bool,
    ) {
        if !(aim_held || second_aim_held) {
            self.reset_spray_state();
            return;
        }

        if cfg.toggle_state.get("Spray Mode") {
            if cfg.toggle_state.get("Cursor Check") {
                if let Some((bx, by, bw, bh)) = detection_box {
                    if !cursor_in_box {
                        if self.is_spraying {
                            self.release_mouse_button();
                        }
                        return;
                    }
                }
            }
            if !self.is_spraying {
                self.hold_mouse_button();
            }
            return;
        }

        // Single click logic.
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCK)
            .unwrap_or_default()
            .as_millis() as u64;
        let trigger_delay_ms = (cfg.slider_settings.get("Auto Trigger Delay") * 1000.0) as u64;
        if let Some(last) = self.last_click {
            if now - last < trigger_delay_ms {
                return;
            }
        }

        let (down, up) = self.get_mouse_actions();
        down();
        std::thread::sleep(std::time::Duration::from_millis(20));
        up();
        self.last_click = Some(now);
    }

    fn get_mouse_actions(&self) -> (Box<dyn Fn()>, Box<dyn Fn()>) {
        match self.method {
            MouseMethod::ViGEmBus => {
                let vigem = self.vigem.clone();
                (Box::new(move || vigem.send_click(1)), Box::new(move || vigem.send_click(0)))
            }
            MouseMethod::Makcu => {
                let makcu = self.makcu.clone();
                (
                    Box::new(move || makcu.as_ref().map(|p| p.send_click(1))),
                    Box::new(move || makcu.as_ref().map(|p| p.send_click(0))),
                )
            }
            MouseMethod::Arduino => {
                let arduino = self.arduino.clone();
                (
                    Box::new(move || arduino.as_ref().map(|p| p.send_mouse_command(0, 0, 1))),
                    Box::new(move || arduino.as_ref().map(|p| p.send_mouse_command(0, 0, 0))),
                )
            }
        }
    }

    fn hold_mouse_button(&mut self) {
        if self.is_spraying {
            return;
        }
        let (down, _) = self.get_mouse_actions();
        down();
        self.is_spraying = true;
    }

    fn release_mouse_button(&mut self) {
        if !self.is_spraying {
            return;
        }
        let (_, up) = self.get_mouse_actions();
        up();
        self.is_spraying = false;
    }
}

impl Drop for MouseManager {
    fn drop(&mut self) {
        self.disconnect_arduino();
        self.disconnect_makcu();
    }
}
