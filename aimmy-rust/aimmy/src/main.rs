use aimmy_ui::AimmyApp;

fn main() {
    // Aimmy is a graphical desktop app, primarily targeting Windows.
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Aimmy 2.5.0 (Rust)")
            .with_inner_size([1280.0, 720.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Aimmy 2.5.0 (Rust)",
        native_options,
        Box::new(|_cc| Ok(Box::new(AimmyApp::new()))),
    );
}
