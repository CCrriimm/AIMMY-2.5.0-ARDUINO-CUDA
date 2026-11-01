use aimmy_core::types::{DisplayInfo, RectangleI};
use std::sync::{Arc, Mutex};

#[cfg(target_os = "windows")]
use windows::Win32::Foundation::{HWND, RECT};
#[cfg(target_os = "windows")]
use windows::Win32::Graphics::Gdi::{
    EnumDisplayMonitors, GetMonitorInfoW, MONITORINFOEXW, MONITORINFOF_PRIMARY,
};
#[cfg(target_os = "windows")]
use std::mem;

/// Event published when the display configuration changes.
#[derive(Debug, Clone)]
pub struct DisplayChangedEvent {
    pub display_index: i32,
    pub display_info: DisplayInfo,
    pub bounds: RectangleI,
    pub working_area: RectangleI,
}

/// Central manager for multi-display functionality.
///
/// Mirrors the C# `DisplayManager`. On Windows it enumerates monitors via
/// `EnumDisplayMonitors`; on other platforms it reports a single synthetic
/// display so the rest of the codebase stays compilable.
pub struct DisplayManager {
    inner: Arc<Mutex<DisplayManagerInner>>,
}

struct DisplayManagerInner {
    displays: Vec<DisplayInfo>,
    current_index: usize,
    current: Option<DisplayInfo>,
    initialized: bool,
    /// Subscribers notified on display change.
    subscribers: Vec<Box<dyn Fn(DisplayChangedEvent) + Send + Sync>>,
}

impl DisplayManager {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(DisplayManagerInner {
                displays: Vec::new(),
                current_index: 0,
                current: None,
                initialized: false,
                subscribers: Vec::new(),
            })),
        }
    }

    pub fn with_subscriber<F>(mut self, f: F) -> Self
    where
        F: Fn(DisplayChangedEvent) + Send + Sync + 'static,
    {
        let mut guard = self.inner.lock().unwrap();
        guard.subscribers.push(Box::new(f));
        self
    }

    pub fn initialize(&self) {
        let mut guard = self.inner.lock().unwrap();
        if guard.initialized {
            return;
        }
        Self::refresh_displays_inner(&mut *guard);
        Self::load_saved_display_inner(&mut *guard);
        guard.initialized = true;
        drop(guard);
        Self::notify_subscribers(&self.inner);
    }

    /// Re-read the list of physical displays.
    pub fn refresh_displays(&self) {
        let mut guard = self.inner.lock().unwrap();
        let old_count = guard.displays.len();
        let old_index = guard.current_index;
        Self::refresh_displays_inner(&mut *guard);

        if guard.displays.len() != old_count || guard.current_index != old_index {
            drop(guard);
            Self::notify_subscribers(&self.inner);
        }
    }

    /// Set the active display by index.
    pub fn set_display(&self, index: usize) -> bool {
        let mut guard = self.inner.lock().unwrap();
        if index >= guard.displays.len() {
            return false;
        }
        let old_index = guard.current_index;
        guard.current_index = index;
        guard.current = guard.displays.get(index).cloned();
        drop(guard);
        if old_index != index {
            Self::notify_subscribers(&self.inner);
        }
        true
    }

    pub fn current_display(&self) -> Option<DisplayInfo> {
        self.inner.lock().unwrap().current.clone()
    }

    pub fn current_display_index(&self) -> usize {
        self.inner.lock().unwrap().current_index
    }

    pub fn screen_width(&self) -> i32 {
        self.inner
            .lock()
            .unwrap()
            .current
            .as_ref()
            .map(|d| d.bounds.width())
            .unwrap_or(0)
    }

    pub fn screen_height(&self) -> i32 {
        self.inner
            .lock()
            .unwrap()
            .current
            .as_ref()
            .map(|d| d.bounds.height())
            .unwrap_or(0)
    }

    pub fn screen_left(&self) -> i32 {
        self.inner
            .lock()
            .unwrap()
            .current
            .as_ref()
            .map(|d| d.bounds.left)
            .unwrap_or(0)
    }

    pub fn screen_top(&self) -> i32 {
        self.inner
            .lock()
            .unwrap()
            .current
            .as_ref()
            .map(|d| d.bounds.top)
            .unwrap_or(0)
    }

    pub fn working_area(&self) -> RectangleI {
        self.inner
            .lock()
            .unwrap()
            .current
            .as_ref()
            .map(|d| d.working_area.clone())
            .unwrap_or(RectangleI::new(0, 0, 0, 0))
    }

    pub fn display_count(&self) -> usize {
        self.inner.lock().unwrap().displays.len()
    }

    pub fn get_all_displays(&self) -> Vec<DisplayInfo> {
        self.inner.lock().unwrap().displays.clone()
    }

    pub fn get_display(&self, index: usize) -> Option<DisplayInfo> {
        self.inner.lock().unwrap().displays.get(index).cloned()
    }

    /// Check if a screen point lies within the current display bounds.
    pub fn is_point_in_current_display(&self, px: i32, py: i32) -> bool {
        self.inner
            .lock()
            .unwrap()
            .current
            .as_ref()
            .map(|d| d.contains(px, py))
            .unwrap_or(false)
    }

    /// Convert screen coordinates to the current display's coordinate system.
    pub fn screen_to_display(&self, px: i32, py: i32) -> (i32, i32) {
        let guard = self.inner.lock().unwrap();
        let d = guard.current.as_ref().unwrap();
        (px - d.bounds.left, py - d.bounds.top)
    }

    pub fn force_refresh(&self) {
        let mut guard = self.inner.lock().unwrap();
        Self::refresh_displays_inner(&mut *guard);
        Self::load_saved_display_inner(&mut *guard);
        drop(guard);
        Self::notify_subscribers(&self.inner);
    }

    // ---------- internals ----------

    fn refresh_displays_inner(guard: &mut DisplayManagerInner) {
        #[cfg(target_os = "windows")]
        {
            guard.displays.clear();
            let mut index: i32 = 0;
            unsafe {
                let callback = |_hmonitor: isize, _hdc: isize, rect: *const RECT, data: *mut std::ffi::c_void| -> i32 {
                    let ctx = unsafe { &mut *(data as *mut MonitorEnumCtx) };
                    let r = unsafe { rect.read() };
                    let mut mi = MONITORINFOEXW::default();
                    mi.s.monitor_info.cbSize = mem::size_of::<MONITORINFOEXW>() as u32;
                    if unsafe { GetMonitorInfoW(r.hMonitor, &mut mi as *mut _ as *mut _).as_bool() } {
                        let bounds = RectangleI::new(
                            r.left,
                            r.top,
                            r.right,
                            r.bottom,
                        );
                        let working = RectangleI::new(
                            mi.s.monitor_info.rcWork.left,
                            mi.s.monitor_info.rcWork.top,
                            mi.s.monitor_info.rcWork.right,
                            mi.s.monitor_info.rcWork.bottom,
                        );
                        let device_name = String::from_utf16_lossy(
                            &mi.szDevice[..mi.s.monitor_info.cbSize as usize / 2],
                        )
                        .trim_end_matches('\0')
                        .to_string();
                        ctx.displays.push(DisplayInfo {
                            index,
                            is_primary: (mi.s.monitor_info.dwFlags & MONITORINFOF_PRIMARY.0) != 0,
                            device_name,
                            bounds,
                            working_area: working,
                        });
                        ctx.index += 1;
                    }
                    1 // continue enumeration
                };
                let mut ctx = MonitorEnumCtx {
                    index,
                    displays: Vec::new(),
                };
                let cb: extern "system" fn(isize, isize, *const RECT, *mut std::ffi::c_void) -> i32 =
                    callback;
                let _ = unsafe {
                    EnumDisplayMonitors(
                        HWND::default(),
                        None,
                        Some(cb),
                        &mut ctx as *mut _ as *mut std::ffi::c_void,
                    )
                };
                guard.displays = ctx.displays;
            }
            index = guard.displays.len() as i32;
        }

        #[cfg(not(target_os = "windows"))]
        {
            guard.displays = vec![DisplayInfo {
                index: 0,
                is_primary: true,
                device_name: "Primary Display".into(),
                bounds: RectangleI::new(0, 0, 1920, 1080),
                working_area: RectangleI::new(0, 0, 1920, 1080),
            }];
        }

        // Clamp current index if displays were removed.
        if guard.current_index >= guard.displays.len() {
            guard.current_index = guard
                .displays
                .iter()
                .position(|d| d.is_primary)
                .unwrap_or(0);
        }
        guard.current = guard.displays.get(guard.current_index).cloned();
    }

    fn load_saved_display_inner(guard: &mut DisplayManagerInner) {
        if guard.displays.is_empty() {
            Self::refresh_displays_inner(guard);
        }
        // Prefer the primary display as default.
        if let Some(pos) = guard.displays.iter().position(|d| d.is_primary) {
            guard.current_index = pos;
        } else if !guard.displays.is_empty() {
            guard.current_index = 0;
        }
        guard.current = guard.displays.get(guard.current_index).cloned();
    }

    fn notify_subscribers(inner: &Arc<Mutex<DisplayManagerInner>>) {
        let guard = inner.lock().unwrap();
        let event = guard.current.as_ref().map(|d| DisplayChangedEvent {
            display_index: guard.current_index as i32,
            display_info: d.clone(),
            bounds: d.bounds.clone(),
            working_area: d.working_area.clone(),
        });
        drop(guard);
        if let Some(event) = event {
            // Re-lock is not needed; we call subscribers outside the lock.
            let guard = inner.lock().unwrap();
            for sub in &guard.subscribers {
                sub(event.clone());
            }
        }
    }

    pub fn dispose(&self) {
        let mut guard = self.inner.lock().unwrap();
        guard.initialized = false;
        guard.subscribers.clear();
    }
}

struct MonitorEnumCtx {
    index: i32,
    displays: Vec<DisplayInfo>,
}

impl Default for DisplayManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DisplayManager {
    fn drop(&mut self) {
        self.dispose();
    }
}
