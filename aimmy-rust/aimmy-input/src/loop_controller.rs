/// Closed-loop stability controller for the aim loop.
///
/// Port of Aimmy's `AimLoopController`. It learns how far the aim loop can
/// push each update without overshooting, for any game/optic/sensitivity
/// without knowing the pixels-per-count conversion factor.
pub struct AimLoopController {
    adaptive_enabled: bool,
    keep_after_overshoot: f64,
    recovery: f64,

    gain: f64,
    min_gain: f64,
    min_keep: f64,
    max_keep: f64,

    tolerance_px: f64,
    dead_zone_px: f64,
    max_tolerance_px: f64,
    tolerance_decay_per_second: f64,
    chatter_window_ms: i64,
    last_overshoot_ms: i64,

    pending_err_x: f64,
    pending_err_y: f64,
    pending_cmd_x: f64,
    pending_cmd_y: f64,
    has_pending: bool,

    clean_ticks: i32,
    last_tick_ms: i64,
    last_dt_scale: f64,
}

const REFERENCE_DT_MS: f64 = 16.0;
const MIN_DT_SCALE: f64 = 0.70;
const MAX_DT_SCALE: f64 = 2.00;
const MIN_COMMAND_COUNTS: f64 = 0.5;
const MIN_GAIN: f64 = 0.03;
const MAX_GAIN: f64 = 1.0;
const MIN_KEEP: f64 = 0.15;
const MAX_KEEP: f64 = 0.85;
const RECOVERY_PER_SECOND: f64 = 0.9;
const CLEAN_TICKS_BEFORE_RECOVERY: i32 = 4;
const MAX_CLEAN_TICKS: i32 = 1000;
const DEAD_ZONE_PX: f64 = 3.0;
const MAX_TOLERANCE_PX: f64 = 36.0;
const TOLERANCE_DECAY_PER_SECOND: f64 = 2.0;
const CHATTER_WINDOW_MS: i64 = 200;

impl Default for AimLoopController {
    fn default() -> Self {
        Self::new()
    }
}

impl AimLoopController {
    pub fn new() -> Self {
        Self {
            adaptive_enabled: true,
            keep_after_overshoot: 0.5,
            recovery: 0.35,
            gain: 1.0,
            min_gain: MIN_GAIN,
            min_keep: MIN_KEEP,
            max_keep: MAX_KEEP,
            tolerance_px: DEAD_ZONE_PX,
            dead_zone_px: DEAD_ZONE_PX,
            max_tolerance_px: MAX_TOLERANCE_PX,
            tolerance_decay_per_second: TOLERANCE_DECAY_PER_SECOND,
            chatter_window_ms: CHATTER_WINDOW_MS,
            last_overshoot_ms: i64::MIN,
            pending_err_x: 0.0,
            pending_err_y: 0.0,
            pending_cmd_x: 0.0,
            pending_cmd_y: 0.0,
            has_pending: false,
            clean_ticks: 0,
            last_tick_ms: -1,
            last_dt_scale: 1.0,
        }
    }

    pub fn is_adaptive_enabled(&self) -> bool {
        self.adaptive_enabled
    }

    pub fn current_gain(&self) -> f64 {
        self.gain
    }

    pub fn lock_tolerance_pixels(&self) -> f64 {
        if self.adaptive_enabled {
            self.tolerance_px
        } else {
            self.dead_zone_px
        }
    }

    pub fn configure(&mut self, adaptive: bool, damping: f64, recovery: f64) {
        self.adaptive_enabled = adaptive;
        // "Damping": how much movement is taken away after an overshoot.
        self.keep_after_overshoot = (1.0 - damping).clamp(self.min_keep, self.max_keep);
        self.recovery = recovery.clamp(0.05, 1.0);
    }

    /// Feed one measured target error (screen px relative to crosshair).
    pub fn evaluate(&mut self, err_x: f64, err_y: f64) {
        let now = time_ms();
        let dt_ms = if self.last_tick_ms < 0 {
            REFERENCE_DT_MS
        } else {
            (now - self.last_tick_ms).clamp(0.5, 100.0) as f64
        };
        self.last_tick_ms = now;
        self.last_dt_scale = (dt_ms / REFERENCE_DT_MS).clamp(MIN_DT_SCALE, MAX_DT_SCALE);

        let magnitude = (err_x * err_x + err_y * err_y).sqrt();

        if self.has_pending {
            let cmd_mag = (self.pending_cmd_x * self.pending_cmd_x + self.pending_cmd_y * self.pending_cmd_y).sqrt();
            if cmd_mag >= MIN_COMMAND_COUNTS {
                let inv = 1.0 / cmd_mag;
                let dir_x = self.pending_cmd_x * inv;
                let dir_y = self.pending_cmd_y * inv;
                let error_before = self.pending_err_x * dir_x + self.pending_err_y * dir_y;
                let error_now = err_x * dir_x + err_y * dir_y;

                if error_before > self.tolerance_px && error_now < -self.tolerance_px {
                    self.gain = (self.gain * self.keep_after_overshoot).max(self.min_gain);
                    if self.last_overshoot_ms != i64::MIN && now - self.last_overshoot_ms <= self.chatter_window_ms {
                        self.tolerance_px = self
                            .tolerance_px
                            .max(error_now.abs())
                            .min(self.max_tolerance_px);
                    }
                    self.last_overshoot_ms = now;
                    self.clean_ticks = 0;
                } else if self.clean_ticks < MAX_CLEAN_TICKS {
                    self.clean_ticks += 1;
                }
            }
            self.has_pending = false;
        }

        if self.clean_ticks >= CLEAN_TICKS_BEFORE_RECOVERY || magnitude < self.tolerance_px {
            self.gain = (self.gain + RECOVERY_PER_SECOND * self.recovery * (dt_ms / 1000.0)).min(MAX_GAIN);
        }

        if magnitude < self.tolerance_px {
            self.tolerance_px = (self.tolerance_px - self.tolerance_decay_per_second * (dt_ms / 1000.0)).max(self.dead_zone_px);
        }
    }

    /// Fraction of the remaining error this update may try to remove.
    pub fn effective_loop_fraction(&self, sensitivity_setting: f64) -> f64 {
        let fraction = (1.0 - sensitivity_setting).clamp(0.02, 1.0) * self.gain;
        fraction.clamp(0.01, 0.95)
    }

    pub fn step_scale(&self) -> f64 {
        self.last_dt_scale
    }

    pub fn record_command(&mut self, err_x: f64, err_y: f64, cmd_x: f64, cmd_y: f64) {
        self.pending_err_x = err_x;
        self.pending_err_y = err_y;
        self.pending_cmd_x = cmd_x;
        self.pending_cmd_y = cmd_y;
        self.has_pending = true;
    }

    pub fn reset(&mut self) {
        self.pending_err_x = 0.0;
        self.pending_err_y = 0.0;
        self.pending_cmd_x = 0.0;
        self.pending_cmd_y = 0.0;
        self.has_pending = false;
        self.clean_ticks = 0;
        self.last_tick_ms = -1;
        self.last_dt_scale = 1.0;
    }
}

fn time_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCK)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
