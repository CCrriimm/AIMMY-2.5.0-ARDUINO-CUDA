/// Exponential Moving Average predictor with velocity tracking
/// (port of Aimmy's `WiseTheFoxPrediction`).
pub struct EmaPredictor {
    alpha: f64,
    ema_x: f64,
    ema_y: f64,
    vel_x: f64,
    vel_y: f64,
    prev_x: f64,
    prev_y: f64,
    last_update: std::time::Instant,
    initialized: bool,
}

const ALPHA: f64 = 0.5;

impl Default for EmaPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl EmaPredictor {
    pub fn new() -> Self {
        Self {
            alpha: ALPHA,
            ema_x: 0.0,
            ema_y: 0.0,
            vel_x: 0.0,
            vel_y: 0.0,
            prev_x: 0.0,
            prev_y: 0.0,
            last_update: std::time::Instant::now(),
            initialized: false,
        }
    }

    pub fn update(&mut self, mx: f64, my: f64) {
        let now = std::time::Instant::now();
        if !self.initialized {
            self.ema_x = mx;
            self.ema_y = my;
            self.prev_x = mx;
            self.prev_y = my;
            self.vel_x = 0.0;
            self.vel_y = 0.0;
            self.last_update = now;
            self.initialized = true;
            return;
        }

        let dt = now.duration_since(self.last_update).as_secs_f64();
        let dt = dt.clamp(0.001, 0.1);

        self.ema_x = self.alpha * mx + (1.0 - self.alpha) * self.ema_x;
        self.ema_y = self.alpha * my + (1.0 - self.alpha) * self.ema_y;

        let new_vel_x = (self.ema_x - self.prev_x) / dt;
        let new_vel_y = (self.ema_y - self.prev_y) / dt;

        self.vel_x = self.alpha * new_vel_x + (1.0 - self.alpha) * self.vel_x;
        self.vel_y = self.alpha * new_vel_y + (1.0 - self.alpha) * self.vel_y;

        self.prev_x = self.ema_x;
        self.prev_y = self.ema_y;
        self.last_update = now;
    }

    /// Estimated position after `lead_time` seconds.
    pub fn estimate(&self, lead_time: f64) -> (f64, f64) {
        (
            self.ema_x + self.vel_x * lead_time,
            self.ema_y + self.vel_y * lead_time,
        )
    }

    pub fn reset(&mut self) {
        self.ema_x = 0.0;
        self.ema_y = 0.0;
        self.vel_x = 0.0;
        self.vel_y = 0.0;
        self.prev_x = 0.0;
        self.prev_y = 0.0;
        self.initialized = false;
    }
}
