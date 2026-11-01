use aimmy_core::logging::LogManager;
use aimmy_core::types::RectangleI;
use std::ptr::null_mut;

/// Selected screen capture method (mirrors the C# dropdown).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureMethod {
    DirectX,
    GdiPlus,
}

impl Default for CaptureMethod {
    fn default() -> Self {
        CaptureMethod::DirectX
    }
}

impl std::fmt::Display for CaptureMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CaptureMethod::DirectX => write!(f, "DirectX"),
            CaptureMethod::GdiPlus => write!(f, "GDI+"),
        }
    }
}

impl From<&str> for CaptureMethod {
    fn from(s: &str) -> Self {
        match s {
            "GDI+" => CaptureMethod::GdiPlus,
            _ => CaptureMethod::DirectX,
        }
    }
}

/// Result of a single screen grab: raw BGRA8 pixels plus the size.
pub struct ScreenGrabResult {
    pub pixels: Vec<u8>, // BGRA8, row stride = width * 4
    pub width: i32,
    pub height: i32,
}

/// Screen capture manager.
///
/// Mirrors the C# `CaptureManager`: DXGI Desktop Duplication as the fast path,
/// GDI+ (`CopyFromScreen`) as the fallback, with automatic failover and
/// display-change handling.
pub struct CaptureManager {
    method: CaptureMethod,
    dxgi: Option<Box<dyn DxgiBackend>>,
    gdi: Option<Box<dyn GdiBackend>>,
    direct_x_failed_permanently: bool,
    consecutive_failures: i32,
    max_consecutive_failures: i32,
    display_changed: bool,
}

const MAX_CONSECUTIVE_FAILURES: i32 = 5;

trait DxgiBackend {
    fn capture(&mut self, box_: &RectangleI) -> Option<ScreenGrabResult>;
}

trait GdiBackend {
    fn capture(&mut self, box_: &RectangleI) -> Option<ScreenGrabResult>;
}

impl CaptureManager {
    pub fn new() -> Self {
        Self {
            method: CaptureMethod::DirectX,
            dxgi: None,
            gdi: None,
            direct_x_failed_permanently: false,
            consecutive_failures: 0,
            max_consecutive_failures: MAX_CONSECUTIVE_FAILURES,
            display_changed: false,
        }
    }

    /// Initialise DXGI duplication for the current display.
    pub fn initialize_dxgi(&mut self) -> std::io::Result<()> {
        self.dxgi = None;
        #[cfg(target_os = "windows")]
        {
            match crate::dxgi_backend::DxgiCapture::new() {
                Ok(cap) => {
                    self.dxgi = Some(Box::new(cap));
                    self.consecutive_failures = 0;
                    LogManager::info("DirectX Desktop Duplication initialized successfully.");
                    Ok(())
                }
                Err(e) => {
                    LogManager::warn(&format!("DirectX Desktop Duplication failed: {}", e));
                    self.direct_x_failed_permanently = true;
                    self.method = CaptureMethod::GdiPlus;
                    LogManager::info(
                        "DirectX Desktop Duplication not supported. Switched to GDI+ capture.",
                    );
                    Err(e)
                }
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            self.method = CaptureMethod::GdiPlus;
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "DXGI capture is Windows-only; falling back to GDI+ stub.",
            ))
        }
    }

    /// Handle a pending display re-configuration (re-init DXGI).
    pub fn handle_pending_display_changes(&mut self) {
        if !self.display_changed {
            return;
        }
        let _ = self.initialize_dxgi();
        self.display_changed = false;
    }

    pub fn mark_display_changed(&mut self) {
        self.display_changed = true;
        self.consecutive_failures = 0;
    }

    /// Capture one frame inside `detection_box` (virtual-screen coords).
    pub fn screen_grab(&mut self, detection_box: &RectangleI) -> Option<ScreenGrabResult> {
        if self.method == CaptureMethod::DirectX && self.direct_x_failed_permanently {
            self.method = CaptureMethod::GdiPlus;
        }

        let result = match self.method {
            CaptureMethod::DirectX => {
                if self.dxgi.is_none() {
                    let _ = self.initialize_dxgi();
                }
                self.dxgi.as_mut()?.capture(detection_box)
            }
            CaptureMethod::GdiPlus => {
                if self.gdi.is_none() {
                    self.gdi = Some(Box::new(crate::gdi_backend::GdiCapture));
                }
                self.gdi.as_mut()?.capture(detection_box)
            }
        };

        match result {
            Some(frame) => {
                self.consecutive_failures = 0;
                Some(frame)
            }
            None => {
                self.consecutive_failures += 1;
                if self.consecutive_failures >= self.max_consecutive_failures {
                    self.display_changed = true;
                }
                None
            }
        }
    }

    pub fn capture_method(&self) -> CaptureMethod {
        self.method
    }

    pub fn set_capture_method(&mut self, method: CaptureMethod) {
        if self.method != method {
            self.dxgi = None;
            self.gdi = None;
            self.method = method;
            if method == CaptureMethod::GdiPlus {
                self.dxgi = None;
            } else {
                let _ = self.initialize_dxgi();
            }
        }
    }

    pub fn dispose(&mut self) {
        self.dxgi = None;
        self.gdi = None;
    }
}

impl Drop for CaptureManager {
    fn drop(&mut self) {
        self.dispose();
    }
}

// ===========================================================================
// Windows DXGI backend
// ===========================================================================

#[cfg(target_os = "windows")]
pub mod dxgi_backend {
    use super::*;
    use windows::Win32::Foundation::{HWND, RECT};
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::Graphics::Direct3D11::*;
    use windows::Win32::Graphics::Dxgi::Common::*;

    pub struct DxgiCapture {
        device: ID3D11Device,
        context: ID3D11DeviceContext,
        duplication: IDXGIOutputDuplication,
        staging: Option<ID3D11Texture2D>,
    }

    impl DxgiCapture {
        pub fn new() -> std::io::Result<Self> {
            unsafe {
                let factory: IDXGIFactory1 = CreateDXGIFactory1()?;

                // Find the adapter/output that drives the primary desktop.
                let mut target_output: Option<IDXGIOutput1> = None;
                let mut target_adapter: Option<IDXGIAdapter1> = None;
                for adapter_index in 0u32.. {
                    let adapter = match unsafe { factory.EnumAdapters1(adapter_index) } {
                        Ok(a) => a,
                        Err(_) => break,
                    };
                    for output_index in 0u32.. {
                        let output = match unsafe { adapter.EnumOutputs(output_index) } {
                            Ok(o) => o,
                            Err(_) => break,
                        };
                        let output1: IDXGIOutput1 = unsafe { output.QueryInterface() };
                        let desc = unsafe { output1.Desc() };
                        let desktop = desc.DesktopCoordinates;
                        // Accept the first output that reports a real desktop rect.
                        if desktop.right > desktop.left && desktop.bottom > desktop.top {
                            target_output = Some(output1);
                            target_adapter = Some(adapter);
                            break;
                        }
                        let _ = output;
                    }
                    if target_output.is_some() {
                        break;
                    }
                    let _ = adapter;
                }

                let (adapter, output1) = match (target_adapter, target_output) {
                    (Some(a), Some(o)) => (a, o),
                    _ => {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            "No suitable display output for DirectX capture",
                        ))
                    }
                };

                let feature_levels = [
                    D3D_FEATURE_LEVEL_12_2,
                    D3D_FEATURE_LEVEL_12_1,
                    D3D_FEATURE_LEVEL_12_0,
                    D3D_FEATURE_LEVEL_11_1,
                    D3D_FEATURE_LEVEL_11_0,
                    D3D_FEATURE_LEVEL_10_1,
                    D3D_FEATURE_LEVEL_10_0,
                ];

                let mut device: Option<ID3D11Device> = None;
                let mut context: Option<ID3D11DeviceContext> = None;
                let mut feature_level = D3D_FEATURE_LEVEL_11_0;
                let hr = unsafe {
                    D3D11CreateDevice(
                        Some(&adapter),
                        D3D_DRIVER_TYPE_UNKNOWN,
                        None,
                        0,
                        feature_levels.as_ptr(),
                        D3D11_SDK_VERSION,
                        &mut device,
                        &mut feature_level,
                        &mut context,
                    )
                };
                if hr.is_err() || device.is_none() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Failed to create D3D11 device: {}", hr),
                    ));
                }

                let device = device.unwrap();
                let context = context.unwrap();
                let duplication = unsafe { output1.DuplicateOutput(&device) }?;

                Ok(Self {
                    device,
                    context,
                    duplication,
                    staging: None,
                })
            }
        }

        pub fn capture(&mut self, box_: &RectangleI) -> Option<ScreenGrabResult> {
            unsafe {
                let w = box_.width();
                let h = box_.height();
                if w <= 0 || h <= 0 {
                    return None;
                }

                // Recreate the staging texture when the requested size changes.
                let need_new = self.staging.is_none()
                    || self
                        .staging
                        .as_ref()
                        .and_then(|t| t.Desc().ok())
                        .map(|d| (d.Width as i32, d.Height as i32))
                        != Some((w, h));
                if need_new {
                    self.staging = None;
                    let desc = D3D11_TEXTURE2D_DESC {
                        Width: w as u32,
                        Height: h as u32,
                        MipLevels: 1,
                        ArraySize: 1,
                        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Usage: D3D11_USAGE_STAGING,
                        BindFlags: 0,
                        CPUAccessFlags: D3D11_CPU_ACCESS_READ,
                        MiscFlags: 0,
                    };
                    self.staging = unsafe { self.device.CreateTexture2D(&desc, None).ok() };
                }
                let staging = self.staging.as_ref()?;

                // Frame descriptor (opaque, zero-initialised).
                let mut frame_desc: DXGI_FRAME_DESC = std::mem::zeroed();
                let mut desktop_resource: Option<IDXGIResource> = None;

                let timeout = if self.consecutive_failures > 0 { 5u32 } else { 15u32 };
                let result = unsafe {
                    self.duplication
                        .AcquireNextFrame(timeout, &mut frame_desc, &mut desktop_resource)
                };

                if result == DXGI_ERROR_WAIT_TIMEOUT {
                    // No new frame yet; do not recycle a stale frame (prevents oscillation).
                    return None;
                }
                if result == S_OK && frame_desc.AccumulatedFrames == 0 {
                    let _ = unsafe { self.duplication.ReleaseFrame() };
                    return None;
                }
                if result == DXGI_ERROR_DEVICE_REMOVED || result == DXGI_ERROR_ACCESS_LOST {
                    return None;
                }
                if result.is_err() {
                    return None;
                }

                let screen_texture: ID3D11Texture2D =
                    match desktop_resource { Some(r) => unsafe { r.QueryInterface() }, None => return None };

                let tex_desc = unsafe { screen_texture.Desc().ok()? };
                let tex_width = tex_desc.Width as i32;
                let tex_height = tex_desc.Height as i32;

                // Convert absolute screen coords into the texture's coordinate space,
                // clamped to the texture dimensions.
                let src_left = box_.left.clamp(0, tex_width);
                let src_top = box_.top.clamp(0, tex_height);
                let src_right = box_.right.clamp(0, tex_width);
                let src_bottom = box_.bottom.clamp(0, tex_height);

                if src_right > src_left && src_bottom > src_top {
                    let region = D3D11_BOX {
                        left: src_left as u32,
                        right: src_right as u32,
                        top: src_top as u32,
                        bottom: src_bottom as u32,
                        front: 0,
                        back: 1,
                    };
                    let dst_x = box_.left.max(0) as u32;
                    let dst_y = box_.top.max(0) as u32;
                    unsafe {
                        self.context.CopySubresourceRegion(
                            staging,
                            0,
                            dst_x,
                            dst_y,
                            0,
                            &screen_texture,
                            0,
                            &region,
                        );
                    }
                } else {
                    let _ = unsafe { self.duplication.ReleaseFrame() };
                    return None;
                }

                let _ = desktop_resource;
                let _ = unsafe { self.duplication.ReleaseFrame() };

                let map = unsafe { self.context.Map(staging, 0, D3D11_MAP_READ, 0).ok()? };
                let mut pixels = vec![0u8; (w as usize * h as usize * 4)];
                let dst_stride = w as isize * 4;
                let src_stride = map.RowPitch as isize;
                let copy_bytes = src_stride.min(dst_stride) as usize;

                unsafe {
                    let src = map.pData as *const u8;
                    for y in 0..h {
                        let src_off = (y as isize) * src_stride;
                        let dst_off = (y as usize) * dst_stride as usize;
                        std::ptr::copy_nonoverlapping(
                            src.add(src_off),
                            pixels.as_mut_ptr().add(dst_off),
                            copy_bytes,
                        );
                    }
                    self.context.Unmap(staging, 0);
                }

                Some(ScreenGrabResult {
                    pixels,
                    width: w,
                    height: h,
                })
            }
        }
    }

    impl Drop for DxgiCapture {
        fn drop(&mut self) {
            unsafe {
                let _ = self.duplication.ReleaseFrame();
            }
        }
    }

    // DXGI_FRAME_DESC is defined by the `windows` crate bindings; we rely on it
    // being `Copy`-able and zero-initialisable via `std::mem::zeroed()`.
}

// DXGI stub for non-Windows platforms.
#[cfg(not(target_os = "windows"))]
pub mod dxgi_backend {
    use super::*;
    pub struct DxgiCapture;
    impl DxgiCapture {
        pub fn new() -> std::io::Result<Self> {
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "DXGI unavailable on this platform",
            ))
        }
        pub fn capture(&mut self, _: &RectangleI) -> Option<ScreenGrabResult> {
            None
        }
    }
}

// ===========================================================================
// Windows GDI+ backend
// ===========================================================================

#[cfg(target_os = "windows")]
pub mod gdi_backend {
    use super::*;
    use windows::Win32::Foundation::HWND;
    use windows::Win32::Graphics::Gdi::*;

    pub struct GdiCapture;

    impl GdiBackend for GdiCapture {
        fn capture(&mut self, box_: &RectangleI) -> Option<ScreenGrabResult> {
            unsafe {
                let w = box_.width();
                let h = box_.height();
                if w <= 0 || h <= 0 {
                    return None;
                }
                let hdc_screen = GetDC(HWND::default());
                if hdc_screen.is_invalid() {
                    return None;
                }
                let hdc_mem = CreateCompatibleDC(hdc_screen);
                let hbmp = CreateCompatibleBitmap(hdc_screen, w, h);
                let _old = SelectObject(hdc_mem, hbmp);

                let ok = BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, box_.left, box_.top, SRCCOPY);
                if !ok.as_bool() {
                    let _ = DeleteDC(hdc_mem);
                    let _ = DeleteObject(hbmp);
                    let _ = ReleaseDC(HWND::default(), hdc_screen);
                    return None;
                }

                let mut bmi = BITMAPINFO {
                    bmiHeader: BITMAPINFOHEADER {
                        biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                        biWidth: w,
                        biHeight: -h, // top-down DIB
                        biPlanes: 1,
                        biBitCount: 32,
                        biCompression: BI_RGB,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                let mut pixels = vec![0u8; (w as usize * h as usize * 4)];
                let got = unsafe {
                    GetDIBits(
                        hdc_mem,
                        hbmp,
                        0,
                        h as u32,
                        pixels.as_mut_ptr() as *mut std::ffi::c_void,
                        &mut bmi,
                        0, // DIB_RGB_COLORS
                    )
                };

                let _ = DeleteDC(hdc_mem);
                let _ = DeleteObject(hbmp);
                let _ = ReleaseDC(HWND::default(), hdc_screen);

                if got == 0 {
                    return None;
                }
                Some(ScreenGrabResult {
                    pixels,
                    width: w,
                    height: h,
                })
            }
        }
    }
}

// GDI stub for non-Windows platforms.
#[cfg(not(target_os = "windows"))]
pub mod gdi_backend {
    use super::*;
    pub struct GdiCapture;
    impl GdiBackend for GdiCapture {
        fn capture(&mut self, _: &RectangleI) -> Option<ScreenGrabResult> {
            None
        }
    }
}
