using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using Nefarius.ViGEm.Client;
using Nefarius.ViGEm.Client.Targets;
using Nefarius.ViGEm.Client.Targets.Xbox360;

namespace MouseMovementLibraries.ViGEmSupport
{
    internal static class ViGEmInput
    {
        private static ViGEmClient? _client;
        private static IXbox360Controller? _controller;
        private static bool _isInitialized = false;
        private static bool _initializationFailed = false;
        private static readonly object _lock = new();

        private static short _lastAxisX = 0;
        private static short _lastAxisY = 0;
        private static long _lastMoveTicks = 0;
        private static Timer? _watchdogTimer;

        [DllImport("user32.dll")]
        private static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);

        private const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
        private const uint MOUSEEVENTF_LEFTUP = 0x0004;

        public static bool Initialize()
        {
            if (_isInitialized) return true;
            if (_initializationFailed) return false;

            lock (_lock)
            {
                if (_isInitialized) return true;
                if (_initializationFailed) return false;

                try
                {
                    _client = new ViGEmClient();
                    _controller = _client.CreateXbox360Controller();
                    _controller.Connect();
                    _isInitialized = true;

                    // Watchdog: auto-center stick if no move received for 80ms
                    _watchdogTimer = new Timer(WatchdogCallback, null, 100, 50);

                    return true;
                }
                catch (Exception ex)
                {
                    _initializationFailed = true;
                    System.Windows.MessageBox.Show(
                        $"Failed to initialize ViGEmBus controller emulation:\n{ex.Message}\n\nPlease ensure ViGEmBus v1.22 driver is installed and the system service is running.",
                        "ViGEmBus Error",
                        System.Windows.MessageBoxButton.OK,
                        System.Windows.MessageBoxImage.Warning);
                    return false;
                }
            }
        }

        private static void WatchdogCallback(object? state)
        {
            if (!_isInitialized || _controller == null) return;

            if (_lastAxisX != 0 || _lastAxisY != 0)
            {
                long elapsedMs = (Stopwatch.GetTimestamp() - _lastMoveTicks) * 1000 / Stopwatch.Frequency;
                if (elapsedMs > 80)
                {
                    ResetMovement();
                }
            }
        }

        public static void Move(int deltaX, int deltaY)
        {
            if (!Initialize() || _controller == null) return;

            // Map pixel delta to Xbox 360 analog stick range (-32768 to 32767)
            // Screen delta is typically in range [-150, 150]. Scale factor 250 gives full stick deflection at ~131px
            short axisX = (short)Math.Clamp(deltaX * 250, -32767, 32767);
            // On screen, positive Y is DOWN. On Xbox controller right stick, positive Y is UP.
            // Invert Y so positive screen delta (down) deflects stick down (negative Y).
            short axisY = (short)Math.Clamp(-deltaY * 250, -32767, 32767);

            try
            {
                _controller.SetAxisValue(Xbox360Axis.RightThumbX, axisX);
                _controller.SetAxisValue(Xbox360Axis.RightThumbY, axisY);
                _controller.SubmitReport();

                _lastAxisX = axisX;
                _lastAxisY = axisY;
                _lastMoveTicks = Stopwatch.GetTimestamp();
            }
            catch { }
        }

        public static void ResetMovement()
        {
            if (!_isInitialized || _controller == null) return;

            try
            {
                _controller.SetAxisValue(Xbox360Axis.RightThumbX, 0);
                _controller.SetAxisValue(Xbox360Axis.RightThumbY, 0);
                _controller.SubmitReport();

                _lastAxisX = 0;
                _lastAxisY = 0;
            }
            catch { }
        }

        public static void SendClick(int click)
        {
            if (!Initialize() || _controller == null)
            {
                if (click > 0)
                    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, UIntPtr.Zero);
                else
                    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, UIntPtr.Zero);
                return;
            }

            try
            {
                // Set Right Trigger (RT) for controller-based games
                _controller.SetSliderValue(Xbox360Slider.RightTrigger, click > 0 ? (byte)255 : (byte)0);
                _controller.SubmitReport();

                // Also trigger standard mouse click for games that expect mouse clicks
                if (click > 0)
                    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, UIntPtr.Zero);
                else
                    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, UIntPtr.Zero);
            }
            catch { }
        }

        public static void Dispose()
        {
            lock (_lock)
            {
                try
                {
                    _watchdogTimer?.Dispose();
                    _watchdogTimer = null;

                    ResetMovement();
                    _controller?.Disconnect();
                    _controller = null;
                    _client?.Dispose();
                    _client = null;

                    _isInitialized = false;
                    _initializationFailed = false;
                }
                catch { }
            }
        }
    }
}
