use aimmy_core::types::{Prediction, RectangleF};
use std::collections::HashMap;

/// Detected raw box before FOV/sticky filtering.
pub struct RawBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_id: i32,
}

/// How the ONNX model lays out its output tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// [1, 4 + NUM_CLASSES, NUM_DETECTIONS] — YOLOv8/v9/v11 raw channels-first.
    YoloRawChannelsFirst,
    /// [1, NUM_DETECTIONS, 4 + NUM_CLASSES] — transposed raw.
    YoloRawChannelsLast,
    /// [1, N, 6] (x1, y1, x2, y2, score, class) — YOLOv10/v26 end-to-end.
    YoloEndToEndBoxesFirst,
    /// [1, 6, N] — end-to-end channels-first.
    YoloEndToEndChannelsFirst,
}

/// Parse a rank-3 detection tensor into `Prediction`s filtered by confidence,
/// target class and the FOV window.
///
/// `tensor` is a flat f32 slice with shape `[dim0, dim1, dim2]`.
pub fn parse_output(
    tensor: &[f32],
    dims: [usize; 3],
    format: ModelFormat,
    image_size: f32,
    fov_min: f32,
    fov_max: f32,
    min_confidence: f32,
    classes: &HashMap<i32, String>,
    selected_class: Option<i32>,
    box_left: f32,
    box_top: f32,
) -> Vec<Prediction> {
    let dim0 = dims[0];
    let dim1 = dims[1];
    let dim2 = dims[2];
    let mut predictions: Vec<Prediction> = Vec::with_capacity(dim1.min(dim2));

    // Index into the flat [dim0][dim1][dim2] tensor.
    let at = |t: &[f32], b: usize, j: usize, k: usize| -> f32 {
        t[b * dim1 * dim2 + j * dim2 + k]
    };

    match format {
        ModelFormat::YoloEndToEndBoxesFirst => {
            // [1, N, 6] = (x1, y1, x2, y2, score, class)
            for i in 0..dim1 {
                let x1 = at(tensor, 0, i, 0);
                let y1 = at(tensor, 0, i, 1);
                let x2 = at(tensor, 0, i, 2);
                let y2 = at(tensor, 0, i, 3);
                let conf = at(tensor, 0, i, 4);
                let cid = at(tensor, 0, i, 5) as i32;
                if let Some(p) = box_to_prediction(
                    x1, y1, x2, y2, conf, cid, image_size, fov_min, fov_max,
                    min_confidence, classes, selected_class, box_left, box_top,
                ) {
                    predictions.push(p);
                }
            }
        }
        ModelFormat::YoloEndToEndChannelsFirst => {
            // [1, 6, N]
            for i in 0..dim2 {
                let x1 = at(tensor, 0, 0, i);
                let y1 = at(tensor, 0, 1, i);
                let x2 = at(tensor, 0, 2, i);
                let y2 = at(tensor, 0, 3, i);
                let conf = at(tensor, 0, 4, i);
                let cid = at(tensor, 0, 5, i) as i32;
                if let Some(p) = box_to_prediction(
                    x1, y1, x2, y2, conf, cid, image_size, fov_min, fov_max,
                    min_confidence, classes, selected_class, box_left, box_top,
                ) {
                    predictions.push(p);
                }
            }
        }
        ModelFormat::YoloRawChannelsFirst => {
            // [1, 4 + C, N]
            let num_classes = dim2.saturating_sub(4);
            for i in 0..dim2 {
                let x_center = at(tensor, 0, 0, i);
                let y_center = at(tensor, 0, 1, i);
                let width = at(tensor, 0, 2, i);
                let height = at(tensor, 0, 3, i);
                let (conf, cid) = pick_class(tensor, 0, i, 4, num_classes, dim1, dim2, selected_class);
                if let Some(p) = cxwh_to_prediction(
                    x_center, y_center, width, height, conf, cid, image_size,
                    fov_min, fov_max, min_confidence, classes, box_left, box_top,
                ) {
                    predictions.push(p);
                }
            }
        }
        ModelFormat::YoloRawChannelsLast => {
            // [1, N, 4 + C]
            let num_classes = dim1.saturating_sub(4);
            for i in 0..dim1 {
                let x_center = at(tensor, 0, i, 0);
                let y_center = at(tensor, 0, i, 1);
                let width = at(tensor, 0, i, 2);
                let height = at(tensor, 0, i, 3);
                let (conf, cid) = pick_class(tensor, 0, i, 4, num_classes, dim1, dim2, selected_class);
                if let Some(p) = cxwh_to_prediction(
                    x_center, y_center, width, height, conf, cid, image_size,
                    fov_min, fov_max, min_confidence, classes, box_left, box_top,
                ) {
                    predictions.push(p);
                }
            }
        }
    }

    predictions
}

/// Pick the best class confidence for a single detection.
///
/// `base` is the index of the first class-confidence slot for this detection;
    `stride` is the distance between consecutive class slots in the flat tensor.
fn pick_class(
    tensor: &[f32],
    b: usize,
    i: usize,
    class_offset: usize,
    num_classes: usize,
    dim1: usize,
    dim2: usize,
    selected: Option<i32>,
) -> (f32, i32) {
    let at = |j: usize, k: usize| -> f32 {
        tensor[b * dim1 * dim2 + j * dim2 + k]
    };
    if num_classes <= 1 {
        return (at(4, i), 0);
    }
    if let Some(sel) = selected {
        let conf = if num_classes == dim2 {
            // channels-first: [b, 4 + sel, i]
            at(4 + sel as usize, i)
        } else {
            // channels-last: [b, i, 4 + sel]
            at(i, 4 + sel as usize)
        };
        return (conf, sel);
    }
    let mut best_conf = f32::NEG_INFINITY;
    let mut best_id = 0;
    for c in 0..num_classes {
        let conf = if num_classes == dim2 {
            at(4 + c, i)
        } else {
            at(i, 4 + c)
        };
        if conf > best_conf {
            best_conf = conf;
            best_id = c;
        }
    }
    (best_conf, best_id as i32)
}

fn box_to_prediction(
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    confidence: f32,
    class_id: i32,
    image_size: f32,
    fov_min: f32,
    fov_max: f32,
    min_confidence: f32,
    classes: &HashMap<i32, String>,
    selected_class: Option<i32>,
    box_left: f32,
    box_top: f32,
) -> Option<Prediction> {
    if confidence < min_confidence {
        return None;
    }
    if let Some(sel) = selected_class {
        if class_id != sel {
            return None;
        }
    }
    // Normalised [0..1] coords -> rescale to image_size.
    let (x1, y1, x2, y2) = if x2 <= 1.0 && y2 <= 1.0 && (x2 > 0.0 || y2 > 0.0) {
        (x1 * image_size, y1 * image_size, x2 * image_size, y2 * image_size)
    } else {
        (x1, y1, x2, y2)
    };
    let width = x2 - x1;
    let height = y2 - y1;
    if width <= 0.0 || height <= 0.0 {
        return None;
    }
    if x1 < fov_min || x2 > fov_max || y1 < fov_min || y2 > fov_max {
        return None;
    }
    let cx = x1 + width / 2.0;
    let cy = y1 + height / 2.0;
    let class_name = classes
        .get(&class_id)
        .cloned()
        .unwrap_or_else(|| format!("Class_{class_id}"));
    Some(Prediction {
        rectangle: RectangleF::new(x1, y1, width, height),
        confidence,
        class_id,
        class_name,
        center_x_translated: cx / image_size,
        center_y_translated: cy / image_size,
        screen_center_x: box_left + cx,
        screen_center_y: box_top + cy,
    })
}

fn cxwh_to_prediction(
    x_center: f32,
    y_center: f32,
    width: f32,
    height: f32,
    confidence: f32,
    class_id: i32,
    image_size: f32,
    fov_min: f32,
    fov_max: f32,
    min_confidence: f32,
    classes: &HashMap<i32, String>,
    box_left: f32,
    box_top: f32,
) -> Option<Prediction> {
    if confidence < min_confidence {
        return None;
    }
    let x_min = x_center - width / 2.0;
    let y_min = y_center - height / 2.0;
    let x_max = x_center + width / 2.0;
    let y_max = y_center + height / 2.0;
    if x_min < fov_min || x_max > fov_max || y_min < fov_min || y_max > fov_max {
        return None;
    }
    let class_name = classes
        .get(&class_id)
        .cloned()
        .unwrap_or_else(|| format!("Class_{class_id}"));
    Some(Prediction {
        rectangle: RectangleF::new(x_min, y_min, width, height),
        confidence,
        class_id,
        class_name,
        center_x_translated: x_center / image_size,
        center_y_translated: y_center / image_size,
        screen_center_x: box_left + x_center,
        screen_center_y: box_top + y_center,
    })
}
