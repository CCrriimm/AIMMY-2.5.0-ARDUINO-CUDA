use crate::predictors::PredictionMethod;

/// Single-target Kalman filter (port of Aimmy's `KalmanPrediction`).
///
/// State: [x, y, vx, vy]. A simplified diagonal covariance is used; velocity
/// is estimated from the measurement residual.
pub struct KalmanFilter {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    p00: f64,
    p11: f64,
    p22: f64,
    p33: f64,
    last_update: std::time::Instant,
    initialized: bool,
}

const PROCESS_NOISE: f64 = 0.1;
const MEASUREMENT_NOISE: f64 = 0.5;
const MAX_VELOCITY: f64 = 5000.0;

impl Default for KalmanFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanFilter {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            vx: 0.0,
            vy: 0.0,
            p00: 1.0,
            p11: 1.0,
            p22: 1.0,
            p33: 1.0,
            last_update: std::time::Instant::now(),
            initialized: false,
        }
    }

    pub fn update(&mut self, mx: f64, my: f64) {
        let now = std::time::Instant::now();
        if !self.initialized {
            self.x = mx;
            self.y = my;
            self.vx = 0.0;
            self.vy = 0.0;
            self.last_update = now;
            self.initialized = true;
            return;
        }

        let dt = now.duration_since(self.last_update).as_secs_f64();
        let dt = dt.clamp(0.001, 0.1);

        // Predict step.
        let pred_x = self.x + self.vx * dt;
        let pred_y = self.y + self.vy * dt;

        self.p00 += PROCESS_NOISE;
        self.p11 += PROCESS_NOISE;
        self.p22 += PROCESS_NOISE * 10.0;
        self.p33 += PROCESS_NOISE * 10.0;

        let innovation_x = mx - pred_x;
        let innovation_y = my - pred_y;

        let k = self.p00 / (self.p00 + MEASUREMENT_NOISE);

        self.x = pred_x + k * innovation_x;
        self.y = pred_y + k * innovation_y;
        self.vx += k * innovation_x / dt;
        self.vy += k * innovation_y / dt;

        self.vx = self.vx.clamp(-MAX_VELOCITY, MAX_VELOCITY);
        self.vy = self.vy.clamp(-MAX_VELOCITY, MAX_VELOCITY);

        self.p00 *= 1.0 - k;
        self.p11 *= 1.0 - k;
        self.last_update = now;
    }

    /// Predict where the target will be after `lead_time` seconds.
    pub fn predict(&self, lead_time: f64) -> (f64, f64) {
        let dt = self.last_update.elapsed().as_secs_f64();
        let current_x = self.x + self.vx * dt;
        let current_y = self.y + self.vy * dt;
        (current_x + self.vx * lead_time, current_y + self.vy * lead_time)
    }

    pub fn reset(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
        self.vx = 0.0;
        self.vy = 0.0;
        self.p00 = 1.0;
        self.p11 = 1.0;
        self.p22 = 1.0;
        self.p33 = 1.0;
        self.initialized = false;
    }
}

impl PredictionMethod {
    pub fn name(&self) -> &'static str {
        match self {
            PredictionMethod::KalmanFilter => "Kalman Filter",
            PredictionMethod::Shalloe => "Shall0e's Prediction",
            PredictionMethod::WiseTheFoxEma => "wisethef0x's EMA Prediction",
        }
    }
}
