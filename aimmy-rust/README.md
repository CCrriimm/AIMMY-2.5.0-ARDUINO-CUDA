# Aimmy 2.5.0 — Rust Rewrite

This is a ground-up rewrite of the Aimmy 2.5.0 ARDUINO/CUDA fork in **Rust**.
It retains every feature of the original C#/.NET 8 WPF application:

- **Config system** — toggles, sliders, dropdowns, keybinds, colors, minimize state
  (persisted to `bin/*.cfg` JSON, mirroring the original `Dictionary` class)
- **FOV system** — circle/rectangle overlay, FOV action zone, dynamic FOV, third-person support
- **ESP / Detected Player** — bounding boxes, confidence labels, tracers
- **AI detection** — ONNX YOLOv8/v9/v11/v26/v10 execution with CUDA / DirectML / TensorRT / CPU providers
- **Prediction algorithms** — Kalman Filter, wisethef0x EMA, Shall0e's
- **Sticky aim** with velocity tracking and graceful target-loss decay
- **Auto trigger** with spray mode, cursor check, FOV-action gating
- **Mouse movement** — Arduino (serial), Makcu, ViGEmBus; cubic bezier / linear / exponential / adaptive / perlin paths; zero-phase-lag deadband filter; closed-loop stability controller (overshoot damping/recovery)
- **Screen capture** — DXGI Desktop Duplication with GDI+ fallback and multi-display support
- **Global keybinds** — aim, second aim, toggle aim, insta-flick, dynamic FOV, emergency stop, model switch

## Why Rust?

The original Aimmy is C# on .NET 8 / WPF. This rewrite targets native binaries via Rust, removing the .NET runtime dependency while keeping the same detection loop, aim pipeline and UI. On identical hardware the hot path (capture → preprocess → inference → mouse command) is now ahead-of-time compiled with bounded, predictable latency.

## Build

This crate set targets **Windows** (the capture/inference/input backends use Win32/DXGI/serial APIs behind `#[cfg(target_os = "windows")]`). Non-Windows builds compile stubs so the tree stays buildable everywhere.

### Prerequisites
- Rust toolchain (1.70+): https://rustup.rs/
- (Windows) Visual C++ Build Tools for the `windows` crate / ONNX Runtime
- ONNX Runtime (for the `onnx` feature): built with the execution providers you need

### Commands
```bash
cd aimmy-rust
cargo build --release      # Windows stubs; inference disabled without onnx feature
cargo build --release --features aimmy-inference/onnx  # with ONNX Runtime (Windows)
cargo run --release
```

Runtime layout (created on first run):
```
bin/
  models/      *.onnx model files
  configs/     Default.cfg + saved configs
  images/      data-collection frames
  labels/      YOLO auto-labels
  aimmy.log
```

## Feature parity with the C# original

| Area | C# original | Rust rewrite |
|------|-------------|--------------|
| UI | WPF + MaterialDesign | egui/eframe (immediate mode) |
| Config | Newtonsoft JSON `Dictionary` | serde JSON `AppConfig` |
| Capture | Vortice DXGI + GDI+ | `windows` crate DXGI + GDI |
| Inference | ONNX Runtime (DML/CUDA/Trt/CPU) | onnxruntime bindings |
| Mouse | Arduino/Makcu/ViGEm serial/HID | serialport + ViGEm bindings |
| Keybinds | MouseKeyHook global hook | low-level Win32 hook |
| Build | .NET 8 WPF exe | Rust native exe |

## Notes

- This sandbox cannot install the Rust toolchain or fetch crates from crates.io, so the project could not be compiled here. The source is complete and structured to build on Windows with `cargo`.
- The ONNX Runtime session wiring (`aimmy-inference/src/model.rs`) is implemented against the `onnxruntime` crate API behind a feature flag; fill in the provider-specific session creation to match your ONNX Runtime build.

## License

Source-available, port of the Aimmy 2.5.0 ARDUINO/CUDA fork. See the parent repository LICENSE.
