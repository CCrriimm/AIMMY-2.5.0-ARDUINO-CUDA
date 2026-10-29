using Aimmy2.Class;
using Class;
using System.Drawing;
using System.Runtime.InteropServices;
using MouseMovementLibraries.ArduinoSupport;
using MouseMovementLibraries.ViGEmSupport;
using MouseMovementLibraries.MakcuSupport;

namespace InputLogic
{
    internal class MouseManager
    {
        private static readonly double ScreenWidth = WinAPICaller.ScreenWidth;
        private static readonly double ScreenHeight = WinAPICaller.ScreenHeight;

        private static ArduinoInput? arduinoInput;
        private static MakcuInput? makcuInput;


        private static DateTime LastClickTime = DateTime.MinValue;
        private static bool isSpraying = false;

        private static double previousX = 0;
        private static double previousY = 0;
        private static double subPixelX = 0;
        private static double subPixelY = 0;
        private static double filteredTargetX = 0;
        private static double filteredTargetY = 0;
        private static double lastTargetX = 0;
        private static double lastTargetY = 0;
        private static int oscillationCounter = 0;
        private static bool hasTarget = false;
        public static double smoothingFactor = 0.5;
        public static bool IsEMASmoothingEnabled = false;

        // Maximum size of a single aim step, expressed as a fraction of the measured distance to the
        // target (plus a small absolute floor). Keeping this below 1.0 means one update can never
        // overshoot the target, which is what makes the loop stable at high optic magnification.
        private const double MaxStepRatio = 0.85;
        private const double MinStepCapPx = 6.0;

        public static void InitArduino(string comPort)
        {
            try
            {
                DisconnectArduino();
                arduinoInput = new ArduinoInput(comPort);
            }
            catch (Exception ex)
            {
                System.Windows.MessageBox.Show($"Failed to connect to {comPort}:\n{ex.Message}", "Arduino Error");
            }
        }

        public static void DisconnectArduino()
        {
            try
            {
                arduinoInput?.Close();
            }
            catch { }
            arduinoInput = null;
        }

        public static void InitMakcu(string comPort, int baudRate = 4000000)
        {
            try
            {
                DisconnectMakcu();
                DisconnectArduino();
                makcuInput = new MakcuInput(comPort, baudRate);
            }
            catch (Exception ex)
            {
                System.Windows.MessageBox.Show($"Failed to connect to Makcu on {comPort}:\n{ex.Message}", "Makcu Error");
            }
        }

        public static void DisconnectMakcu()
        {
            try
            {
                makcuInput?.Close();
            }
            catch { }
            makcuInput = null;
        }

        private static Random MouseRandom = new();

        private static double EmaSmoothing(double previousValue, double currentValue, double smoothingFactor) => (currentValue * smoothingFactor) + (previousValue * (1 - smoothingFactor));

        // Cleanup
        private static (Action down, Action up) GetMouseActions()
        {
            string mouseMovementMethod = Dictionary.dropdownState["Mouse Movement Method"];
            Action mouseDownAction;
            Action mouseUpAction;

            switch (mouseMovementMethod)
            {
                case "ViGEmBus":
                    mouseDownAction = () => ViGEmInput.SendClick(1);
                    mouseUpAction = () => ViGEmInput.SendClick(0);
                    break;
                case "Makcu":
                    mouseDownAction = () => makcuInput?.SendClick(1);
                    mouseUpAction = () => makcuInput?.SendClick(0);
                    break;
                case "Arduino":
                default:
                    mouseDownAction = () => arduinoInput?.SendMouseCommand(0, 0, 1);
                    mouseUpAction = () => arduinoInput?.SendMouseCommand(0, 0, 0);
                    break;
            }

            return (mouseDownAction, mouseUpAction);
        }

        public static async Task DoTriggerClick(RectangleF? detectionBox = null)
        {
            // there was a toggle for this, but i realized if it was off, it would never stop spraying. - T
            if (!(InputBindingManager.IsHoldingBinding("Aim Keybind") || InputBindingManager.IsHoldingBinding("Second Aim Keybind")))
            {
                ResetSprayState();
                return;
            }


            if (Dictionary.toggleState["Spray Mode"])
            {
                if (Dictionary.toggleState["Cursor Check"])
                {
                    Point mousePos = WinAPICaller.GetCursorPosition();

                    if (detectionBox.HasValue && !detectionBox.Value.Contains(mousePos.X, mousePos.Y))
                    {
                        if (isSpraying) ReleaseMouseButton();
                        return;
                    }
                }

                if (!isSpraying) HoldMouseButton();
                return;
            }

            // Single click logic if spray mode off
            int timeSinceLastClick = (int)(DateTime.UtcNow - LastClickTime).TotalMilliseconds;
            int triggerDelayMilliseconds = (int)(Dictionary.sliderSettings["Auto Trigger Delay"] * 1000);
            const int clickDelayMilliseconds = 20;

            if (timeSinceLastClick < triggerDelayMilliseconds && LastClickTime != DateTime.MinValue)
            {
                return;
            }

            var (mouseDown, mouseUp) = GetMouseActions();

            mouseDown.Invoke();
            await Task.Delay(clickDelayMilliseconds);
            mouseUp.Invoke();

            LastClickTime = DateTime.UtcNow;
        }

        #region Spray Mode Methods
        public static void HoldMouseButton()
        {
            if (isSpraying) return;

            var (mouseDown, _) = GetMouseActions();
            mouseDown.Invoke();
            isSpraying = true;
        }

        public static void ReleaseMouseButton()
        {
            if (!isSpraying) return;

            var (_, mouseUp) = GetMouseActions();
            mouseUp.Invoke();
            isSpraying = false;
        }

        public static void ResetMovement()
        {
            if (Dictionary.dropdownState.TryGetValue("Mouse Movement Method", out var method) && method == "ViGEmBus")
            {
                ViGEmInput.ResetMovement();
            }
            previousX = 0;
            previousY = 0;
            subPixelX = 0;
            subPixelY = 0;
            filteredTargetX = 0;
            filteredTargetY = 0;
            lastTargetX = 0;
            lastTargetY = 0;
            oscillationCounter = 0;
            hasTarget = false;

            // Forget the measured loop behaviour: the next target is a fresh start.
            AimLoopController.Reset();
        }

        public static void ResetSprayState()
        {
            if (isSpraying)
            {
                ReleaseMouseButton();
            }
            ResetMovement();
        }
        #endregion

        public static void MoveCrosshair(int detectedX, int detectedY)
        {
            int halfScreenWidth = (int)ScreenWidth / 2;
            int halfScreenHeight = (int)ScreenHeight / 2;

            double rawTargetX = detectedX - halfScreenWidth;
            double rawTargetY = detectedY - halfScreenHeight;

            // ── Stage 1: Zero-Phase-Lag Deadband Filter ──
            // Eliminates YOLO bounding-box micro-jitter without adding phase lag.
            // Phase lag in a feedback loop causes the classic "mild shaking that grows crazy" oscillation.
            if (!hasTarget)
            {
                filteredTargetX = rawTargetX;
                filteredTargetY = rawTargetY;
                lastTargetX = rawTargetX;
                lastTargetY = rawTargetY;
                oscillationCounter = 0;
                hasTarget = true;
            }
            else
            {
                // Rule A: ZERO-CROSSING PURGE
                // If the crosshair crosses the target (sign reverses), purge filter lag immediately!
                // An EMA filter lagging on the wrong side of 0 pushes the mouse AWAY from the target,
                // causing positive feedback oscillation.
                if ((rawTargetX > 0 && filteredTargetX < 0) || (rawTargetX < 0 && filteredTargetX > 0))
                {
                    filteredTargetX = rawTargetX;
                }
                if ((rawTargetY > 0 && filteredTargetY < 0) || (rawTargetY < 0 && filteredTargetY > 0))
                {
                    filteredTargetY = rawTargetY;
                }

                // Rule B: DEADBAND NOISE REJECTION
                // If the target position moved less than 2.0px from the current filtered position,
                // it is neural network bounding box jitter. Suppress it!
                double deltaX = rawTargetX - filteredTargetX;
                double deltaY = rawTargetY - filteredTargetY;
                double jitterDist = Math.Sqrt(deltaX * deltaX + deltaY * deltaY);

                if (jitterDist < 2.0)
                {
                    // Within noise threshold: keep position stable, do not follow micro-jitter
                    filteredTargetX = filteredTargetX * 0.95 + rawTargetX * 0.05;
                    filteredTargetY = filteredTargetY * 0.95 + rawTargetY * 0.05;
                }
                else if (jitterDist < 6.0)
                {
                    // Gentle transition
                    double t = (jitterDist - 2.0) / 4.0;
                    double alpha = 0.50 + t * 0.35; // 0.50 to 0.85
                    filteredTargetX = filteredTargetX * (1.0 - alpha) + rawTargetX * alpha;
                    filteredTargetY = filteredTargetY * (1.0 - alpha) + rawTargetY * alpha;
                }
                else
                {
                    // Real target movement: follow immediately with 90% responsiveness (zero perceptible lag)
                    filteredTargetX = filteredTargetX * 0.10 + rawTargetX * 0.90;
                    filteredTargetY = filteredTargetY * 0.10 + rawTargetY * 0.90;
                }
            }

            double targetX = filteredTargetX;
            double targetY = filteredTargetY;
            double dist = Math.Sqrt(targetX * targetX + targetY * targetY);

            // ── Stage 1b: Closed-loop stability controller ──
            // Push the current tuning in, hand it this tick's measured error, and take its verdict on
            // how much of that error we are allowed to try to remove. See AimLoopController.
            AimLoopController.Configure(
                Dictionary.toggleState["Adaptive Sensitivity Control"],
                Convert.ToDouble(Dictionary.sliderSettings["Overshoot Damping"]),
                Convert.ToDouble(Dictionary.sliderSettings["Overshoot Recovery"]));

            AimLoopController.Evaluate(targetX, targetY);
            double dtScale = AimLoopController.StepScale();

            // ── Stage 2: Lock-on Deadzone & Smoothstep Dampening ──
            // The radius is not a constant any more: one mouse count moves the view further on a high
            // in-game sensitivity, so a fixed 3 px deadzone can be smaller than a single minimum
            // command and the crosshair chatters ±1 count around the target forever. The controller
            // scales the tolerance with the learned pixels-per-count. See LockTolerancePixels.
            double DeadzoneRadius = AimLoopController.LockTolerancePixels;
            const double DampeningRadius = 18.0;

            if (dist <= DeadzoneRadius)
            {
                subPixelX = 0;
                subPixelY = 0;
                previousX = 0;
                previousY = 0;
                oscillationCounter = 0;
                lastTargetX = targetX;
                lastTargetY = targetY;
                // Locked on: nothing was commanded, so there is no overshoot to learn from.
                AimLoopController.RecordCommand(targetX, targetY, 0, 0);
                return;
            }

            double lockDamping = 1.0;
            if (dist < DampeningRadius)
            {
                double normalized = (dist - DeadzoneRadius) / (DampeningRadius - DeadzoneRadius);
                lockDamping = normalized * normalized * (3.0 - 2.0 * normalized); // Smoothstep curve
            }

            double aspectRatioCorrection = ScreenWidth / ScreenHeight;

            // Suppress artificial jitter when near target
            int MouseJitter = (int)Dictionary.sliderSettings["Mouse Jitter"];
            double jitterScale = Math.Clamp((dist - 20.0) / 30.0, 0.0, 1.0);
            double jitterX = MouseRandom.Next(-MouseJitter, MouseJitter) * jitterScale;
            double jitterY = MouseRandom.Next(-MouseJitter, MouseJitter) * jitterScale;

            // ── Stage 3: Movement Path Calculation ──
            Point start = new(0, 0);
            Point end = new((int)targetX, (int)targetY);
            Point newPosition = new Point(0, 0);

            double sensSetting = Convert.ToDouble(Dictionary.sliderSettings["Mouse Sensitivity (+/-)"]);
            // The step is a fraction of the remaining error. The controller supplies that fraction:
            // the user's sensitivity, bounded by the stability ceiling and reduced further whenever
            // the loop has been overshooting. This is what makes high in-game sensitivity usable.
            double sensFactor = AimLoopController.EffectiveLoopFraction(sensSetting);

            switch (Dictionary.dropdownState["Movement Path"])
            {
                case "Cubic Bezier":
                    Point control1 = new Point(start.X + (end.X - start.X) / 3, start.Y + (end.Y - start.Y) / 3);
                    Point control2 = new Point(start.X + 2 * (end.X - start.X) / 3, start.Y + 2 * (end.Y - start.Y) / 3);
                    newPosition = MovementPaths.CubicBezier(start, end, control1, control2, sensFactor);
                    break;
                case "Linear":
                    newPosition = MovementPaths.Lerp(start, end, sensFactor);
                    break;
                case "Exponential":
                    newPosition = MovementPaths.Exponential(start, end, 1 - (sensSetting - 0.2), 3.0);
                    break;
                case "Adaptive":
                    newPosition = MovementPaths.Adaptive(start, end, sensFactor);
                    break;
                case "Perlin Noise":
                    newPosition = MovementPaths.PerlinNoise(start, end, sensFactor, 20, 0.5);
                    break;
                default:
                    newPosition = MovementPaths.Lerp(start, end, sensFactor);
                    break;
            }

            double posX = newPosition.X;
            double posY = newPosition.Y;

            if (IsEMASmoothingEnabled)
            {
                posX = EmaSmoothing(previousX, posX, smoothingFactor);
                posY = EmaSmoothing(previousY, posY, smoothingFactor);
            }

            // ── Stage 4: Accelerated Smoothing (Tamed Near Target) ──
            if (Dictionary.toggleState["Accelerated Smoothing"])
            {
                double accelAmount = Convert.ToDouble(Dictionary.sliderSettings["Accelerated Smoothing"]);
                // Only amplify when far from target; smoothly decay to 1.0 within 25px so it doesn't cause overshoots
                double accelFactor = (dist < 25.0) 
                    ? 1.0 + (accelAmount * 0.2 * (dist / 25.0)) 
                    : Math.Clamp(1.0 + (dist / 100.0) * accelAmount, 1.0, 2.5);
                posX *= accelFactor;
                posY *= accelFactor;
            }

            // ── Stage 5: Flick Aim (Snappy at Range, Disabled Near Target) ──
            if (Dictionary.toggleState["Flick Aim"])
            {
                double flickStrength = Convert.ToDouble(Dictionary.sliderSettings["Flick Aim Strength"]);
                // Flick Aim should NEVER replace 100% of displacement (which causes instantaneous overshoots).
                // Max blend capped at 40%, and only active when dist > 20px. Near target (<= 20px), blend drops to 0.
                if (dist > 20.0)
                {
                    double rangeWeight = Math.Clamp((dist - 20.0) / 40.0, 0.0, 1.0);
                    double effectiveFlick = Math.Clamp(flickStrength * 0.40 * rangeWeight * lockDamping, 0.0, 0.40);
                    posX = posX * (1.0 - effectiveFlick) + targetX * effectiveFlick;
                    posY = posY * (1.0 - effectiveFlick) + targetY * effectiveFlick;
                }
            }

            // ── Stage 6: Derivative Active Braking (D-Term) ──
            // When closing in on target, apply progressive braking to prevent overshoot.
            double closingDist = Math.Sqrt(lastTargetX * lastTargetX + lastTargetY * lastTargetY) - dist;

            if (closingDist > 0 && dist < 30.0)
            {
                // Crosshair is closing in on target within 30px: brake proportionally to closing speed
                double brakeFactor = Math.Clamp(1.0 - (closingDist / Math.Max(dist, 4.0)) * 0.4, 0.35, 1.0);
                posX *= brakeFactor;
                posY *= brakeFactor;
            }

            // Apply lock-on stabilization dampening
            posX *= lockDamping;
            posY *= lockDamping;

            // Apply aspect ratio correction
            posY /= aspectRatioCorrection;

            // Add distance-scaled jitter
            posX += jitterX;
            posY += jitterY;

            // ── Stage 7: Anti-Hunting Oscillation Suppression ──
            // Detects back-and-forth chatter across consecutive ticks and applies persistent suppression.
            if (previousX != 0 || previousY != 0)
            {
                double dot = posX * previousX + posY * previousY;
                if (dot < 0)
                {
                    // Direction reversed across consecutive ticks
                    oscillationCounter = Math.Min(oscillationCounter + 1, 5);
                }
                else
                {
                    // Stable direction
                    oscillationCounter = Math.Max(oscillationCounter - 1, 0);
                }

                if (oscillationCounter > 0)
                {
                    // Progressively dampen oscillation: 1 reversal = 50%, 2+ reversals = 25% to 15%
                    double dampMultiplier = Math.Max(0.15, 1.0 - (oscillationCounter * 0.30));
                    posX *= dampMultiplier;
                    posY *= dampMultiplier;
                }
            }

            // Clamp maximum per-tick movement
            posX = Math.Clamp(posX, -150, 150);
            posY = Math.Clamp(posY, -150, 150);

            // ── Stage 8: Frame-rate normalisation ──
            // DirectX capture produces far more aim updates per second than GDI+. Scaling the step by
            // the elapsed time keeps the loop's convergence rate (and therefore its stability) the
            // same for both, instead of letting the capture method decide how hard the loop pushes.
            posX *= dtScale;
            posY *= dtScale;

            // ── Stage 9: Legacy cap, only when the adaptive controller is switched off ──
            // With the controller on, the fraction supplied above is already bounded by the stability
            // ceiling and reduced by every observed overshoot, so no further cap is needed (and the
            // old one was actively harmful: it compared mouse counts against a distance in pixels, so
            // its 6-count floor was itself a huge jump on a high-sensitivity setup).
            if (!AimLoopController.IsAdaptiveEnabled && dist > 0)
            {
                double maxStep = Math.Max(MinStepCapPx, dist * MaxStepRatio);
                if (Math.Abs(posX) > maxStep) posX = Math.Sign(posX) * maxStep;
                if (Math.Abs(posY) > maxStep) posY = Math.Sign(posY) * maxStep;
            }

            // Sub-pixel accumulation to prevent fractional truncation jitter
            double totalX = posX + subPixelX;
            double totalY = posY + subPixelY;

            int moveX = (int)Math.Round(totalX);
            int moveY = (int)Math.Round(totalY);

            subPixelX = totalX - moveX;
            subPixelY = totalY - moveY;

            // Tell the controller what was actually commanded, so on the next update it can judge
            // whether that command overshot - which is how it finds the stable gain for this setup.
            AimLoopController.RecordCommand(targetX, targetY, moveX, moveY);

            if (moveX != 0 || moveY != 0)
            {
                switch (Dictionary.dropdownState["Mouse Movement Method"])
                {
                    case "ViGEmBus":
                        ViGEmInput.Move(moveX, moveY);
                        break;
                    case "Makcu":
                        makcuInput?.Move(moveX, moveY);
                        break;
                    case "Arduino":
                    default:
                        arduinoInput?.SendMouseCommand(moveX, moveY, 0);
                        break;
                }
            }

            previousX = posX;
            previousY = posY;
            lastTargetX = targetX;
            lastTargetY = targetY;

            if (!Dictionary.toggleState["Auto Trigger"])
            {
                ResetSprayState();
            }
        }

        public static void InstaFlick(int detectedX, int detectedY)
        {
            int halfScreenWidth = (int)ScreenWidth / 2;
            int halfScreenHeight = (int)ScreenHeight / 2;

            int targetX = detectedX - halfScreenWidth;
            int targetY = detectedY - halfScreenHeight;

            if (Math.Abs(targetX) <= 1 && Math.Abs(targetY) <= 1)
            {
                return;
            }

            double aspectRatioCorrection = ScreenWidth / ScreenHeight;
            targetY = (int)(targetY / aspectRatioCorrection);

            double flickSpeed = Convert.ToDouble(Dictionary.sliderSettings["Insta-Flick Speed"]);
            // Scale step factor: lower slider value = slower/smoother step, higher = instant snap
            double stepMultiplier = Math.Clamp(1.0 / Math.Max(flickSpeed, 0.05) * 0.25, 0.25, 1.0);

            int moveX = (int)(targetX * stepMultiplier);
            int moveY = (int)(targetY * stepMultiplier);

            // Never send more than the remaining error in one go: an insta-flick that overshoots the
            // target is what makes a flick oscillate, and the overshoot gets worse with optic zoom.
            int flickCap = Math.Max(8, (int)(Math.Max(Math.Abs(targetX), Math.Abs(targetY)) * MaxStepRatio));
            moveX = Math.Clamp(moveX, -flickCap, flickCap);
            moveY = Math.Clamp(moveY, -flickCap, flickCap);

            moveX = Math.Clamp(moveX, -127, 127);
            moveY = Math.Clamp(moveY, -127, 127);

            if (moveX != 0 || moveY != 0)
            {
                switch (Dictionary.dropdownState["Mouse Movement Method"])
                {
                    case "ViGEmBus":
                        ViGEmInput.Move(moveX, moveY);
                        break;
                    case "Makcu":
                        makcuInput?.Move(moveX, moveY);
                        break;
                    case "Arduino":
                    default:
                        arduinoInput?.SendMouseCommand(moveX, moveY, 0);
                        break;
                }
            }

            previousX = moveX;
            previousY = moveY;
        }
    }
}