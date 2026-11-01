/// Movement path styles (mirrors the C# dropdown).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MovementPath {
    CubicBezier,
    Linear,
    Exponential,
    Adaptive,
    PerlinNoise,
}

impl Default for MovementPath {
    fn default() -> Self {
        MovementPath::CubicBezier
    }
}

impl From<&str> for MovementPath {
    fn from(s: &str) -> Self {
        match s {
            "Linear" => MovementPath::Linear,
            "Exponential" => MovementPath::Exponential,
            "Adaptive" => MovementPath::Adaptive,
            "Perlin Noise" => MovementPath::PerlinNoise,
            _ => MovementPath::CubicBezier,
        }
    }
}

/// Compute one step along a movement path from `start` toward `end`.
/// `t` is the loop-supplied fraction (0..1) of the error to close.
pub fn movement_path(
    path: MovementPath,
    start: (f64, f64),
    end: (f64, f64),
    t: f64,
    sens_setting: f64,
) -> (f64, f64) {
    match path {
        MovementPath::CubicBezier => cubic_bezier(start, end, t),
        MovementPath::Linear => lerp(start, end, t),
        MovementPath::Exponential => exponential(start, end, 1.0 - (sens_setting - 0.2), 3.0),
        MovementPath::Adaptive => adaptive(start, end, t, 100.0),
        MovementPath::PerlinNoise => perlin_noise(start, end, t, 20.0, 0.1),
    }
}

fn cubic_bezier(start: (f64, f64), end: (f64, f64), t: f64) -> (f64, f64) {
    let u = 1.0 - t;
    let tt = t * t;
    let uu = u * u;
    let c1x = start.0 + (end.0 - start.0) / 3.0;
    let c1y = start.1 + (end.1 - start.1) / 3.0;
    let c2x = start.0 + 2.0 * (end.0 - start.0) / 3.0;
    let c2y = start.1 + 2.0 * (end.1 - start.1) / 3.0;
    let x = uu * u * start.0 + 3.0 * uu * t * c1x + 3.0 * u * tt * c2x + tt * t * end.0;
    let y = uu * u * start.1 + 3.0 * uu * t * c1y + 3.0 * u * tt * c2y + tt * t * end.1;
    (x, y)
}

fn lerp(start: (f64, f64), end: (f64, f64), t: f64) -> (f64, f64) {
    (start.0 + (end.0 - start.0) * t, start.1 + (end.1 - start.1) * t)
}

fn exponential(start: (f64, f64), end: (f64, f64), t: f64, exponent: f64) -> (f64, f64) {
    let p = t.powf(exponent);
    (
        start.0 + (end.0 - start.0) * p,
        start.1 + (end.1 - start.1) * p,
    )
}

fn adaptive(start: (f64, f64), end: (f64, f64), t: f64, threshold: f64) -> (f64, f64) {
    let distance = ((end.0 - start.0).powi(2) + (end.1 - start.1).powi(2)).sqrt();
    if distance < threshold {
        lerp(start, end, t)
    } else {
        cubic_bezier(start, end, t)
    }
}

fn perlin_noise(start: (f64, f64), end: (f64, f64), t: f64, amplitude: f64, frequency: f64) -> (f64, f64) {
    let base_x = start.0 + (end.0 - start.0) * t;
    let base_y = start.1 + (end.1 - start.1) * t;
    let noise_x = perlin(t * frequency, 0.0) * amplitude;
    let noise_y = perlin(t * frequency, 100.0) * amplitude;
    let perp_x = -(end.1 - start.1);
    let perp_y = end.0 - start.0;
    let perp_len = (perp_x * perp_x + perp_y * perp_y).sqrt();
    let (perp_x, perp_y) = if perp_len > 0.0 {
        (perp_x / perp_len, perp_y / perp_len)
    } else {
        (0.0, 0.0)
    };
    let final_x = base_x + perp_x * noise_x + noise_y * 0.3;
    let final_y = base_y + perp_y * noise_x + noise_y * 0.3;
    (final_x, final_y)
}

fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp_f(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

fn grad(hash: i32, x: f64, y: f64) -> f64 {
    let h = hash & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        0.0
    };
    let u = if (h & 1) == 0 { u } else { -u };
    let v = if (h & 2) == 0 { v } else { -v };
    u + v
}

fn perlin(x: f64, y: f64) -> f64 {
    let mut permutation = [0i32; 512];
    for i in 0..256 {
        permutation[i] = i as i32;
    }
    // Deterministic shuffle (port of the C# fixed permutation table).
    let mut p = permutation.clone();
    for i in (0..256).rev() {
        let j = (fastrand::u32(0..=i as u32)) as usize;
        p.swap(i, j);
    }
    for i in 0..256 {
        permutation[256 + i] = p[i];
    }

    let xf = x.floor() as i32 & 255;
    let yf = y.floor() as i32 & 255;
    let xr = x - x.floor();
    let yr = y - y.floor();
    let u = fade(xr);
    let v = fade(yr);
    let a = permutation[xf as usize] + yf;
    let aa = permutation[a as usize];
    let ab = permutation[(a + 1) as usize];
    let b = permutation[(xf + 1) as usize] + yf;
    let ba = permutation[b as usize];
    let bb = permutation[(b + 1) as usize];
    lerp_f(
        lerp_f(grad(permutation[aa as usize], xr, yr), grad(permutation[ba as usize], xr - 1.0, yr), u),
        lerp_f(
            grad(permutation[ab as usize], xr, yr - 1.0),
            grad(permutation[bb as usize], xr - 1.0, yr - 1.0),
            u,
        ),
        v,
    )
}
