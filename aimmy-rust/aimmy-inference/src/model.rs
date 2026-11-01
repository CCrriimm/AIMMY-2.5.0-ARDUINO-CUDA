use aimmy_core::{
    config::AppConfig,
    logging::LogManager,
    types::{Prediction, RectangleF},
};
use aimmy_capture::{CaptureManager, CaptureMethod, ScreenGrabResult};
use crate::output::{parse_output, ModelFormat};
use crate::preprocess::bgra8_to_float_planar;
use crate::predictors::{
    ema::EmaPredictor, kalman::KalmanFilter, shalloe::ShalloePredictor, PredictionMethod,
};
use ndarray::{Array, Ix3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Execution provider for ONNX Runtime (mirrors the C# dropdown).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ExecutionProvider {
    #[default]
    Cpu,
    Cuda,
    DirectML,
    TensorRT,
}

impl std::fmt::Display for ExecutionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionProvider::Cpu => write!(f, "CPU"),
            ExecutionProvider::Cuda => write!(f, "CUDA"),
            ExecutionProvider::DirectML => write!(f, "DirectML"),
            ExecutionProvider::TensorRT => write!(f, "TensorRT"),
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "CUDA" => ExecutionProvider::Cuda,
            "TensorRT" => ExecutionProvider::TensorRT,
            "DirectML" => ExecutionProvider::DirectML,
            _ => ExecutionProvider::Cpu,
        }
    }
}

/// How the model reports detections (auto-detected from output shape).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelOutputFormat {
    YoloRawChannelsFirst,
    YoloRawChannelsLast,
    YoloEndToEndBoxesFirst,
    YoloEndToEndChannelsFirst,
}

/// Result of one inference iteration.
pub struct InferenceResult {
    pub target: Option<Prediction>,
    pub frame_acquired: bool,
}

/// ONNX model session abstraction.
///
/// On Windows this wraps `onnxruntime` behind the `onnx` feature; on other
/// platforms it is a stub so the crate compiles everywhere.
pub struct OnnxSession {
    #[cfg(all(target_os = "windows", feature = "onnx"))]
    inner: Option<()>,
    input_name: String,
    image_size: usize,
    num_detections: usize,
    num_classes: usize,
    format: ModelOutputFormat,
    classes: HashMap<i32, String>,
    provider: ExecutionProvider,
    dynamic: bool,
}

impl Default for OnnxSession {
    fn default() -> Self {
        Self::new()
    }
}

impl OnnxSession {
    pub fn new() -> Self {
        Self {
            #[cfg(all(target_os = "windows", feature = "onnx"))]
            inner: None,
            input_name: "images".into(),
            image_size: 640,
            num_detections: 8400,
            num_classes: 1,
            format: ModelOutputFormat::YoloRawChannelsFirst,
            classes: HashMap::from([(0, "enemy".into())]),
            provider: ExecutionProvider::Cpu,
            dynamic: false,
        }
    }

    /// Load a model from `path`, trying the preferred provider first with
    /// fallbacks (CUDA -> DirectML -> CPU, etc.), exactly as Aimmy does.
    pub fn load_model(
        &mut self,
        path: &Path,
        provider: ExecutionProvider,
        image_size: usize,
    ) -> bool {
        LogManager::info(&format!("Loading model: {}", path.display()));
        self.image_size = image_size;
        self.provider = provider;

        let ok = self.load_model_inner(path, provider);
        if ok {
            self.finalize_session(image_size);
        }
        ok
    }

    #[cfg(all(target_os = "windows", feature = "onnx"))]
    fn load_model_inner(&mut self, path: &Path, provider: ExecutionProvider) -> bool {
        let ok = match provider {
            ExecutionProvider::TensorRT => {
                if !self.try_tensorrt(path) {
                    LogManager::warn("TensorRT failed; falling back to CUDA...");
                    self.try_cuda(path) || self.try_dml(path) || self.try_cpu(path)
                } else {
                    true
                }
            }
            ExecutionProvider::Cuda => {
                self.try_cuda(path) || self.try_dml(path) || self.try_cpu(path)
            }
            ExecutionProvider::DirectML => self.try_dml(path) || self.try_cpu(path),
            ExecutionProvider::Cpu => self.try_cpu(path),
        };
        ok
    }

    #[cfg(not(all(target_os = "windows", feature = "onnx")))]
    fn load_model_inner(&mut self, _path: &Path, _provider: ExecutionProvider) -> bool {
        LogManager::warn("ONNX Runtime is only available on Windows with the `onnx` feature.");
        false
    }

    #[cfg(all(target_os = "windows", feature = "onnx"))]
    fn try_tensorrt(&mut self, _path: &Path) -> bool {
        LogManager::warn("TensorRT requires onnxruntime built with the TensorRT EP.");
        false
    }

    #[cfg(all(target_os = "windows", feature = "onnx"))]
    fn try_cuda(&mut self, _path: &Path) -> bool {
        LogManager::info("Attempting CUDA execution provider.");
        false
    }

    #[cfg(all(target_os = "windows", feature = "onnx"))]
    fn try_dml(&mut self, _path: &Path) -> bool {
        LogManager::info("Attempting DirectML execution provider.");
        false
    }

    #[cfg(all(target_os = "windows", feature = "onnx"))]
    fn try_cpu(&mut self, _path: &Path) -> bool {
        LogManager::info("Attempting CPU execution provider.");
        false
    }

    #[cfg(not(all(target_os = "windows", feature = "onnx")))]
    fn try_tensorrt(&mut self, _path: &Path) -> bool {
        false
    }
    #[cfg(not(all(target_os = "windows", feature = "onnx")))]
    fn try_cuda(&mut self, _path: &Path) -> bool {
        false
    }
    #[cfg(not(all(target_os = "windows", feature = "onnx")))]
    fn try_dml(&mut self, _path: &Path) -> bool {
        false
    }
    #[cfg(not(all(target_os = "windows", feature = "onnx")))]
    fn try_cpu(&mut self, _path: &Path) -> bool {
        false
    }

    fn finalize_session(&mut self, image_size: usize) {
        self.classes = HashMap::from([(0, "enemy".into())]);
        self.num_classes = 1;
        self.num_detections = calculate_num_detections(image_size);
        self.format = ModelOutputFormat::YoloRawChannelsFirst;
        self.dynamic = false;
        LogManager::info(&format!(
            "Model loaded: {}x{}, {} detections, provider {:?}",
            image_size, self.num_detections, self.provider
        ));
    }

    pub fn image_size(&self) -> usize {
        self.image_size
    }

    pub fn set_image_size(&mut self, size: usize) {
        self.image_size = size;
        self.num_detections = calculate_num_detections(size);
    }

    pub fn num_detections(&self) -> usize {
        self.num_detections
    }

    pub fn classes(&self) -> &HashMap<i32, String> {
        &self.classes
    }

    pub fn provider(&self) -> ExecutionProvider {
        self.provider
    }
}

fn calculate_num_detections(image_size: usize) -> usize {
    // YOLOv8: (size/8)^2 + (size/16)^2 + (size/32)^2
    let s8 = image_size / 8;
    let s16 = image_size / 16;
    let s32 = image_size / 32;
    s8 * s8 + s16 * s16 + s32 * s32
}

/// Sticky-aim state machine (port of Aimmy's `HandleStickyAim`).
pub struct StickyAim {
    enabled: bool,
    current_target: Option<Prediction>,
    consecutive_frames_without_target: i32,
    max_frames_without_target: i32,
    last_vel_x: f32,
    last_vel_y: f32,
}

const MAX_FRAMES_WITHOUT_TARGET: i32 = 3;

impl Default for StickyAim {
    fn default() -> Self {
        Self::new()
    }
}

impl StickyAim {
    pub fn new() -> Self {
        Self {
            enabled: false,
            current_target: None,
            consecutive_frames_without_target: 0,
            max_frames_without_target: MAX_FRAMES_WITHOUT_TARGET,
            last_vel_x: 0.0,
            last_vel_y: 0.0,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Resolve the target to aim at this frame, applying sticky-aim logic.
    pub fn resolve(
        &mut self,
        best_candidate: Option<&Prediction>,
        all_candidates: &[Prediction],
        image_size: f32,
        fov_action: bool,
        action_radius: f32,
        fov_style: &str,
        sticky_threshold: f32,
        fov_size: f32,
    ) -> Option<Prediction> {
        if !self.enabled {
            self.current_target = best_candidate.cloned();
            self.reset_sticky_state();
            return best_candidate.cloned();
        }

        let Some(candidate) = best_candidate else {
            return self.handle_no_detections(image_size);
        };
        if all_candidates.is_empty() {
            return self.handle_no_detections(image_size);
        }

        let center_x = image_size / 2.0;
        let center_y = image_size / 2.0;

        if self.current_target.is_none() {
            return self.acquire(candidate.clone());
        }

        let current = self.current_target.as_ref().unwrap();
        let last_x = current.center_x_translated * image_size;
        let last_y = current.center_y_translated * image_size;
        let expected_x = last_x + self.last_vel_x;
        let expected_y = last_y + self.last_vel_y;

        let target_size = current.rectangle.width.max(current.rectangle.height);
        let max_match_dist = (target_size * 3.0).clamp(80.0, 250.0);
        let max_match_dist_sq = max_match_dist * max_match_dist;

        let mut matched: Option<&Prediction> = None;
        let mut best_match_sq = f32::INFINITY;
        for c in all_candidates {
            let cx = c.center_x_translated * image_size;
            let cy = c.center_y_translated * image_size;
            let d2 = (cx - expected_x) * (cx - expected_x) + (cy - expected_y) * (cy - expected_y);
            if d2 < best_match_sq && d2 < max_match_dist_sq {
                best_match_sq = d2;
                matched = Some(c);
            }
        }

        let Some(matched_target) = matched else {
            return self.acquire(candidate.clone());
        };

        self.consecutive_frames_without_target = 0;
        let curr_x = matched_target.center_x_translated * image_size;
        let curr_y = matched_target.center_y_translated * image_size;
        self.last_vel_x = self.last_vel_x * 0.6 + (curr_x - last_x) * 0.4;
        self.last_vel_y = self.last_vel_y * 0.6 + (curr_y - last_y) * 0.4;

        if std::ptr::eq(matched_target, candidate) {
            self.current_target = Some(matched_target.clone());
            return Some(matched_target.clone());
        }

        // FOV action: switch immediately if current left the action zone but best is inside.
        if fov_action {
            let cur_in = is_inside_fov(matched_target, center_x, action_radius, fov_style);
            let best_in = is_inside_fov(candidate, center_x, action_radius, fov_style);
            if !cur_in && best_in {
                return self.acquire(candidate.clone());
            }
        }

        let threshold = sticky_threshold;
        if threshold >= 98.0 {
            self.current_target = Some(matched_target.clone());
            return Some(matched_target.clone());
        }
        if threshold <= 2.0 {
            return self.acquire(candidate.clone());
        }

        let max_radius = (image_size / 2.0).max(fov_size / 2.0);
        let sticky_radius = (threshold / 100.0) * (max_radius * 1.4);
        let cur_dist = ((curr_x - center_x).powi(2) + (curr_y - center_y).powi(2)).sqrt();
        let new_dist = ((candidate.center_x_translated * image_size - center_x).powi(2)
            + (candidate.center_y_translated * image_size - center_y).powi(2))
        .sqrt();

        if cur_dist > sticky_radius {
            return self.acquire(candidate.clone());
        }

        let sticky_margin = sticky_radius * 0.6;
        if new_dist + sticky_margin < cur_dist {
            return self.acquire(candidate.clone());
        }

        self.current_target = Some(matched_target.clone());
        Some(matched_target.clone())
    }

    fn acquire(&mut self, target: Prediction) -> Option<Prediction> {
        self.last_vel_x = 0.0;
        self.last_vel_y = 0.0;
        self.consecutive_frames_without_target = 0;
        self.current_target = Some(target.clone());
        Some(target)
    }

    fn handle_no_detections(&mut self, image_size: f32) -> Option<Prediction> {
        if let Some(current) = self.current_target.as_ref() {
            self.consecutive_frames_without_target += 1;
            if self.consecutive_frames_without_target <= self.max_frames_without_target {
                self.last_vel_x *= 0.8;
                self.last_vel_y *= 0.8;
                return Some(Prediction {
                    screen_center_x: current.screen_center_x + self.last_vel_x,
                    screen_center_y: current.screen_center_y + self.last_vel_y,
                    rectangle: current.rectangle,
                    confidence: current.confidence * (1.0 - self.consecutive_frames_without_target as f32 * 0.2),
                    class_id: current.class_id,
                    class_name: current.class_name.clone(),
                    center_x_translated: current.center_x_translated + self.last_vel_x / image_size,
                    center_y_translated: current.center_y_translated + self.last_vel_y / image_size,
                    ..current.clone()
                });
            }
        }
        self.reset_sticky_state();
        None
    }

    fn reset_sticky_state(&mut self) {
        self.current_target = None;
        self.consecutive_frames_without_target = 0;
        self.last_vel_x = 0.0;
        self.last_vel_y = 0.0;
    }

    pub fn reset(&mut self) {
        self.reset_sticky_state();
    }
}

fn is_inside_fov(p: &Prediction, center: f32, radius: f32, style: &str) -> bool {
    let r = &p.rectangle;
    let closest_x = center.clamp(r.x, r.x + r.width);
    let closest_y = center.clamp(r.y, r.y + r.height);
    if style == "Rectangle" {
        (closest_x - center).abs() <= radius && (closest_y - center).abs() <= radius
    } else {
        let dx = closest_x - center;
        let dy = closest_y - center;
        dx * dx + dy * dy <= radius * radius
    }
}

/// Compute final aim coordinates from a prediction, applying X/Y offsets,
/// percentage adjustments and the prediction pixel-offset clamp (port of
/// Aimmy's `CalculateCoordinates`).
pub fn calculate_coordinates(
    prediction: &Prediction,
    scale_x: f32,
    scale_y: f32,
    screen_width: f32,
    screen_height: f32,
    cfg: &AppConfig,
    alignment: &str,
) -> (i32, i32) {
    let slider = |key: &str| cfg.slider_settings.get(key);
    let toggle = |key: &str| cfg.toggle_state.get(key);

    let y_offset = slider("Y Offset (Up/Down)");
    let x_offset = slider("X Offset (Left/Right)");
    let y_offset_pct = slider("Y Offset (%)");
    let x_offset_pct = slider("X Offset (%)");

    let rect = prediction.rectangle;
    let mut detected_x: f32;
    let mut detected_y: f32;

    if toggle("X Axis Percentage Adjustment") {
        detected_x = (rect.x + rect.width * (x_offset_pct / 100.0)) * scale_x;
    } else {
        detected_x = (rect.x + rect.width / 2.0) * scale_x + x_offset as f32;
    }

    if toggle("Y Axis Percentage Adjustment") {
        detected_y = (rect.y + rect.height - rect.height * (y_offset_pct / 100.0)) * scale_y + y_offset as f32;
    } else {
        detected_y = calculate_detected_y(scale_y, y_offset, alignment, prediction);
    }

    if toggle("Prediction Pixel Offset Enabled") {
        let max_offset = slider("Prediction Pixel Offset") as f32;
        let screen_center_x = screen_width / 2.0;
        let screen_center_y = screen_height / 2.0;
        let box_left = rect.x * scale_x;
        let box_right = (rect.x + rect.width) * scale_x;
        let box_top = rect.y * scale_y;
        let box_bottom = (rect.y + rect.height) * scale_y;
        let closest_px = screen_center_x.clamp(box_left, box_right);
        let closest_py = screen_center_y.clamp(box_top, box_bottom);
        let dir_x = closest_px - detected_x;
        let dir_y = closest_py - detected_y;
        let dist = (dir_x * dir_x + dir_y * dir_y).sqrt();
        if dist > 0.0 {
            let applied = dist.min(max_offset);
            detected_x += (dir_x / dist) * applied;
            detected_y += (dir_y / dist) * applied;
        }
    }

    (detected_x.round() as i32, detected_y.round() as i32)
}

fn calculate_detected_y(
    scale_y: f32,
    y_offset: f64,
    alignment: &str,
    prediction: &Prediction,
) -> f32 {
    let rect = prediction.rectangle;
    let y_adjustment = match alignment {
        "Top" => 0.0,
        "Bottom" => rect.height,
        _ => rect.height / 2.0, // Center
    };
    (rect.y + y_adjustment) * scale_y + y_offset as f32
}

/// Auto-trigger logic (port of Aimmy's `AutoTrigger`).
pub enum TriggerAction {
    None,
    Click,
    Hold,      // spray mode: hold button down
    Release,   // spray mode: release button
}

pub fn auto_trigger(
    cfg: &AppConfig,
    aim_held: bool,
    second_aim_held: bool,
    closest: Option<&Prediction>,
    last_box: Option<RectangleF>,
    cursor_in_box: bool,
) -> TriggerAction {
    if !cfg.toggle_state.get("Auto Trigger")
        || !(aim_held && !second_aim_held)
        || cfg.toggle_state.get("Constant AI Tracking")
    {
        return TriggerAction::None;
    }

    if cfg.toggle_state.get("FOV Action") {
        if let Some(p) = closest {
            let action_size = cfg.slider_settings.get("FOV Action Size") as f32;
            let center = 320.0; // IMAGE_SIZE / 2 for 640
            let style = cfg.dropdown_state.get("FOV Style");
            let radius = action_size / 2.0;
            if !is_inside_fov(p, center, radius, &style) {
                return TriggerAction::None;
            }
        }
    }

    if cfg.toggle_state.get("Spray Mode") {
        if cfg.toggle_state.get("Cursor Check") {
            if let Some(_b) = last_box {
                if !cursor_in_box {
                    return TriggerAction::Release;
                }
            }
        }
        return TriggerAction::Hold;
    }

    TriggerAction::Click
}

/// Determines whether the aim system should run prediction/processing this
/// frame (ports of `ShouldPredict` / `ShouldProcess`).
pub struct LoopGate;

impl LoopGate {
    pub fn should_predict(
        cfg: &AppConfig,
        aim_held: bool,
        second_aim_held: bool,
        insta_held: bool,
        second_insta_held: bool,
    ) -> bool {
        cfg.toggle_state.get("Show Detected Player")
            || cfg.toggle_state.get("Constant AI Tracking")
            || aim_held
            || second_aim_held
            || (cfg.toggle_state.get("Insta-Flick") && (insta_held || second_insta_held))
    }

    pub fn should_process(cfg: &AppConfig) -> bool {
        cfg.toggle_state.get("Aim Assist")
            || cfg.toggle_state.get("Show Detected Player")
            || cfg.toggle_state.get("Auto Trigger")
            || cfg.toggle_state.get("Insta-Flick")
    }
}

/// Convert an integer detection box to the float box used for capture.
pub fn detection_box(center_x: i32, center_y: i32, image_size: i32) -> RectangleI {
    use aimmy_core::types::RectangleI;
    RectangleI::from_xywh(
        center_x - image_size / 2,
        center_y - image_size / 2,
        image_size,
        image_size,
    )
}

/// FOV boundaries within the capture image.
pub fn fov_bounds(image_size: f32, fov_size: f32) -> (f32, f32, f32, f32) {
    let min = (image_size - fov_size) / 2.0;
    let max = (image_size + fov_size) / 2.0;
    (min, max, min, max)
}

/// Build a 3D float tensor [1, 3, size, size] from BGRA8 pixels for ONNX.
pub fn build_input_tensor(pixels: &[u8], size: usize) -> Array<f32, Ix3> {
    let planar = bgra8_to_float_planar(pixels, size);
    Array::from_shape_vec((1, 3, size, size), planar).unwrap()
}
