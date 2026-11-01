/// Velocity-based predictor using a short history of velocity samples
/// (port of Aimmy's `ShalloePredictionV2`).
pub struct ShalloePredictor {
    vel_x_history: Vec<f64>,
    vel_y_history: Vec<f64>,
    prev_x: f64,
    prev_y: f64,
    initialized: bool,
}

const MAX_HISTORY: usize = 5;

impl Default for ShalloePredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl ShalloePredictor {
    pub fn new() -> Self {
        Self {
            vel_x_history: Vec::with_capacity(MAX_HISTORY),
            vel_y_history: Vec::with_capacity(MAX_HISTORY),
            prev_x: 0.0,
            prev_y: 0.0,
            initialized: false,
        }
    }

    pub fn update(&mut self, mx: f64, my: f64) {
        if !self.initialized {
            self.prev_x = mx;
            self.prev_y = my;
            self.initialized = true;
            return;
        }

        let vx = mx - self.prev_x;
        let vy = my - self.prev_y;

        if self.vel_x_history.len() >= MAX_HISTORY {
            self.vel_x_history.remove(0);
            self.vel_y_history.remove(0);
        }
        self.vel_x_history.push(vx);
        self.vel_y_history.push(vy);

        self.prev_x = mx;
        self.prev_y = my;
    }

    /// Predicted position after `lead_multiplier` frames.
    pub fn predict(&self, lead_multiplier: f64) -> (f64, f64) {
        if !self.initialized || self.vel_x_history.is_empty() {
            return (self.prev_x, self.prev_y);
        }
        let avg_vx: f64 = self.vel_x_history.iter().sum::<f64>() / self.vel_x_history.len() as f64;
        let avg_vy: f64 = self.vel_y_history.iter().sum::<f64>() / self.vel_y_history.len() as f64;
        (
            self.prev_x + avg_vx * lead_multiplier,
            self.prev_y + avg_vy * lead_multiplier,
        )
    }

    pub fn reset(&mut self) {
        self.vel_x_history.clear();
        self.vel_y_history.clear();
        self.prev_x = 0.0;
        self.prev_y = 0.0;
        self.initialized = false;
    }
}
