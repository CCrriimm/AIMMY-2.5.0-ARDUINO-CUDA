pub mod kalman;
pub mod ema;
pub mod shalloe;

/// Which prediction algorithm is active.
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

impl std::fmt::Display for PredictionMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PredictionMethod::KalmanFilter => write!(f, "Kalman Filter"),
            PredictionMethod::Shalloe => write!(f, "Shall0e's Prediction"),
            PredictionMethod::WiseTheFoxEma => write!(f, "wisethef0x's EMA Prediction"),
        }
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
