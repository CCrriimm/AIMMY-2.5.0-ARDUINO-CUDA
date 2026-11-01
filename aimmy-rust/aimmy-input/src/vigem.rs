/// ViGEm virtual gamepad mouse backend.
///
/// On Windows this wraps the ViGEmClient SDK; elsewhere it is a no-op stub so
/// the crate compiles on any platform.
#[cfg(target_os = "windows")]
pub struct ViGEmInput {
    client: Option<()>, // placeholder for the real ViGEm client handle
}

#[cfg(target_os = "windows")]
impl ViGEmInput {
    pub fn new() -> Self {
        Self { client: None }
    }

    pub fn send_click(&self, _down: i32) {
        // ViGEm click injection would go here.
    }

    pub fn move_(&self, _x: i32, _y: i32) {
        // ViGEm relative mouse movement would go here.
    }

    pub fn reset_movement(&self) {}
    pub fn clone(&self) -> Self {
        Self { client: None }
    }
}

#[cfg(not(target_os = "windows"))]
pub struct ViGEmInput;

#[cfg(not(target_os = "windows"))]
impl ViGEmInput {
    pub fn new() -> Self {
        Self
    }
    pub fn send_click(&self, _down: i32) {}
    pub fn move_(&self, _x: i32, _y: i32) {}
    pub fn reset_movement(&self) {}
    pub fn clone(&self) -> Self {
        Self
    }
}
