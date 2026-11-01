/// Raw Windows input injection (SendInput) for systems without a hardware backend.
///
/// Kept here for completeness; the default Aimmy backends are Arduino/Makcu/ViGEm.
#[cfg(target_os = "windows")]
pub mod raw {
    pub fn move_mouse(x: i32, y: i32) {
        // SendInput with MOUSEINPUT (MOUSEEVENTF_MOVE | MOUSEEVENTF_VIRTUALDESK).
    }
    pub fn click(down: bool) {}
}

#[cfg(not(target_os = "windows"))]
pub mod raw {
    pub fn move_mouse(_x: i32, _y: i32) {}
    pub fn click(_down: bool) {}
}
