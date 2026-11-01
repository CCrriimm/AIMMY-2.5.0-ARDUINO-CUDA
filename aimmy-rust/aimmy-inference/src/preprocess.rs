use aimmy_core::types::RectangleI;

/// Convert BGRA8 pixels (as produced by the capture backends) into a planar
/// float32 NCHW tensor normalised to [0..1], matching Aimmy's
/// `BitmapToFloatArrayInPlace` (BGR channel order, Windows bitmaps are BGR).
///
/// Output layout: [R channel (H*W), G channel (H*W), B channel (H*W)].
pub fn bgra8_to_float_planar(pixels: &[u8], size: usize) -> Vec<f32> {
    let total = size * size;
    let mut out = vec![0.0f32; 3 * total];
    let (r, g, b) = (out.as_mut_ptr(), out.as_mut_ptr().add(total), out.as_mut_ptr().add(2 * total));

    // SAFETY: we write exactly 3*total disjoint f32 slots.
    unsafe {
        for y in 0..size {
            let row = y * size;
            for x in 0..size {
                let i = (row + x) * 4;
                // Windows BMP/DXGI frames are BGRA8.
                let b_val = pixels[i] as f32 / 255.0;
                let g_val = pixels[i + 1] as f32 / 255.0;
                let r_val = pixels[i + 2] as f32 / 255.0;
                *r.add(row + x) = r_val;
                *g.add(row + x) = g_val;
                *b.add(row + x) = b_val;
            }
        }
    }
    out
}
