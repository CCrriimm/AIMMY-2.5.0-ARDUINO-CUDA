using Class;

namespace InputLogic
{
    /// <summary>
    /// Closed-loop stability controller for the aim controller.
    ///
    /// THE PROBLEM
    /// -----------
    /// Aim assist is a feedback loop:
    ///
    ///     capture frame -> detect target -> error = target - crosshair -> move mouse -> game camera moves -> repeat
    ///
    /// The aim code produces a movement in *mouse counts*, but the game decides how many *screen pixels*
    /// one count is worth, using its own in-game sensitivity and the current optic. That conversion is
    /// the plant gain:
    ///
    ///     loopGainPerTick = (fraction of the error removed per update) * pixelsPerCount
    ///
    /// At low in-game sensitivity the product stays below 1 and the crosshair converges smoothly. At
    /// high sensitivity the same counts swing the view several times further, the product passes 1, and
    /// every correction overshoots. The crosshair then alternates either side of the target forever:
    /// the relentless shaking that is only "fixed" by lowering the in-game sensitivity. The old tuning
    /// knobs could not cure it because they change the fraction, not the product.
    ///
    /// WHY THERE IS NO SENSITIVITY MEASUREMENT
    /// ---------------------------------------
    /// The obvious fix is to measure pixels-per-count and cap the step in pixels. That measurement
    /// cannot be trusted: the controller only sees the *combination* of its own movement and the
    /// target's movement, so an enemy strafing is indistinguishable from a too-large mouse step. Trying
    /// to learn it made aim lag behind moving targets badly. This class therefore uses no unit
    /// assumptions at all - it adapts to whatever the loop is doing, which works for any game, optic
    /// or sensitivity without knowing the conversion factor.
    ///
    /// WHAT IT DOES
    /// ------------
    /// 1. SUPPRESSION. Any update that pushes towards the target and comes back past it is an
    ///    overshoot. Each one cuts the allowed movement multiplicatively (halved by default, floor
    ///    0.03), so the loop
    ///    walks itself down to a gain that is stable on this particular setup within a few updates -
    ///    usually 2 or 3, before the shake is even visible.
    ///
    /// 2. RECOVERY. Movement is only given back after a run of clean updates, and only slowly, so the
    ///    controller cannot see-saw between "suppress" and "release" while still oscillating. Without
    ///    this the aim is permanently sluggish after one bad moment.
    ///
    /// 3. QUANTISATION TOLERANCE. Mouse input has a minimum step of one count, and at high sensitivity
    ///    a single count can be worth more pixels than the remaining error. Trying to hold a fixed 3 px
    ///    lock on such a setup is physically impossible: every attempt lands one count past, which is
    ///    corrected by one count back - the last, most stubborn part of the shake. When overshoots
    ///    repeat (genuine chatter, not a flick) the lock-on radius grows to the amplitude actually
    ///    observed. It relaxes back afterwards, so aim does not stay short.
    ///
    /// 4. FRAME-RATE NORMALISATION. The step is scaled by elapsed time, so behaviour is the same at 60
    ///    and at 400 updates per second (DirectX capture produces far more than GDI+).
    ///
    /// Learned values deliberately survive target loss: MouseManager.ResetMovement() runs on every frame
    /// aim is idle, so clearing them there would restart the shake on every single trigger pull.
    /// </summary>
    internal static class AimLoopController
    {
        // ── Tunables, pushed in from the Aim Config sliders every update ──
        private static bool _adaptiveEnabled = true;
        private static double _keepAfterOvershoot = 0.5;
        private static double _recovery = 0.35;

        // ── Learned movement suppression: 1.0 = untouched, floor = very cautious ──
        private static double _gain = 1.0;
        private const double MinGain = 0.03;
        private const double MaxGain = 1.0;
        private const double MinKeepAfterOvershoot = 0.15;
        private const double MaxKeepAfterOvershoot = 0.85;
        private const double RecoveryPerSecond = 0.9;
        private const int CleanTicksBeforeRecovery = 4;
        private const int MaxCleanTicks = 1000;

        // ── Learned lock-on tolerance, in screen pixels ──
        private static double _tolerancePx = DeadZonePx;
        private const double DeadZonePx = 3.0;
        private const double MaxTolerancePx = 36.0;
        private const double ToleranceDecayPerSecond = 2.0;
        private const long ChatterWindowMs = 200;
        private static long _lastOvershootMs = long.MinValue;

        // ── Loop bookkeeping ──
        private const double ReferenceDtMs = 16.0;
        private const double MinDtScale = 0.70;
        private const double MaxDtScale = 2.00;
        private const double MinCommandCounts = 0.5;

        private static double _pendingErrX;
        private static double _pendingErrY;
        private static double _pendingCmdX;
        private static double _pendingCmdY;
        private static bool _hasPending;

        private static int _cleanTicks;
        private static long _lastTickMs = -1;
        private static double _lastDtScale = 1.0;

        public static bool IsAdaptiveEnabled => _adaptiveEnabled;
        public static double CurrentGain => _gain;

        /// <summary>
        /// Radius around the target that counts as "locked on", in screen pixels. Grows when the setup
        /// physically cannot resolve better than one mouse count, relaxes back when it can.
        /// </summary>
        public static double LockTolerancePixels => _adaptiveEnabled ? _tolerancePx : DeadZonePx;

        /// <summary>
        /// Applies the Aim Config sliders. Called once per aim update, so dragging a slider takes
        /// effect on the very next update.
        /// </summary>
        public static void Configure(bool adaptive, double damping, double recovery)
        {
            _adaptiveEnabled = adaptive;

            // "Damping" is how much of the movement is taken away after an overshoot: 0.5 keeps half
            // (halves it), higher cuts harder and settles calmer, lower is snappier but can ring.
            _keepAfterOvershoot = Math.Clamp(1.0 - damping, MinKeepAfterOvershoot, MaxKeepAfterOvershoot);

            _recovery = Math.Clamp(recovery, 0.05, 1.0);
        }

        /// <summary>
        /// Feeds one measured target error (screen pixels, relative to the crosshair) into the
        /// controller. Call once per aim update, before this update's movement is computed.
        /// </summary>
        public static void Evaluate(double errX, double errY)
        {
            long now = Environment.TickCount64;
            double dtMs = _lastTickMs < 0 ? ReferenceDtMs : Math.Clamp(now - _lastTickMs, 0.5, 100.0);
            _lastTickMs = now;

            // Normalised once per update and reused by StepScale(), so the two can never disagree.
            // (StepScale used to read the clock again after Evaluate had just set it, get ~0 ms and
            // pin the scale at its floor, permanently weakening every movement.)
            _lastDtScale = Math.Clamp(dtMs / ReferenceDtMs, MinDtScale, MaxDtScale);

            double magnitude = Math.Sqrt(errX * errX + errY * errY);

            if (_hasPending)
            {
                double cmdMagnitude = Math.Sqrt(_pendingCmdX * _pendingCmdX + _pendingCmdY * _pendingCmdY);

                if (cmdMagnitude >= MinCommandCounts)
                {
                    double inv = 1.0 / cmdMagnitude;
                    double dirX = _pendingCmdX * inv;
                    double dirY = _pendingCmdY * inv;

                    // Error measured along the direction we pushed on the previous update.
                    //   > 0 : still short of the target
                    //   < 0 : we went past it
                    double errorBefore = _pendingErrX * dirX + _pendingErrY * dirY;
                    double errorNow = errX * dirX + errY * dirY;

                    // A real overshoot has to come back *past* the tolerance; merely crossing zero is
                    // within the noise of the detector and of a target that is moving.
                    if (errorBefore > _tolerancePx && errorNow < -_tolerancePx)
                    {
                        _gain = Math.Max(MinGain, _gain * _keepAfterOvershoot);

                        // Widen the lock-on radius only when overshoots repeat - that is genuine
                        // chatter. A single overshoot (a flick, or a target running past the
                        // crosshair) must not leave aim permanently stopping short of the target.
                        if (_lastOvershootMs != long.MinValue && now - _lastOvershootMs <= ChatterWindowMs)
                        {
                            _tolerancePx = Math.Min(MaxTolerancePx, Math.Max(_tolerancePx, Math.Abs(errorNow)));
                        }

                        _lastOvershootMs = now;
                        _cleanTicks = 0;
                    }
                    else if (_cleanTicks < MaxCleanTicks)
                    {
                        _cleanTicks++;
                    }
                }

                _hasPending = false;
            }

            // Give movement back slowly, and only after a run of clean updates (or while locked on,
            // where no command is being issued and nothing can oscillate).
            if (_cleanTicks >= CleanTicksBeforeRecovery || magnitude < _tolerancePx)
            {
                _gain = Math.Min(MaxGain, _gain + RecoveryPerSecond * _recovery * (dtMs / 1000.0));
            }

            // Relax the lock-on radius while calm, so it tracks a sensitivity or optic change instead
            // of staying at whatever it once had to grow to.
            if (magnitude < _tolerancePx)
            {
                _tolerancePx = Math.Max(DeadZonePx, _tolerancePx - ToleranceDecayPerSecond * (dtMs / 1000.0));
            }
        }

        /// <summary>
        /// Fraction of the remaining error this update is allowed to try to remove. Bounded by the
        /// user's stability ceiling so no slider combination can ask for more than the loop can take.
        /// </summary>
        public static double EffectiveLoopFraction(double sensitivitySetting)
        {
            // The user's own sensitivity, reduced by whatever the loop has learned it must hold back.
            // With the controller disabled _gain stays at 1.0, so this is exactly the old behaviour.
            double fraction = Math.Clamp(1.0 - sensitivitySetting, 0.02, 1.0) * _gain;
            return Math.Clamp(fraction, 0.01, 0.95);
        }

        /// <summary>
        /// Records the command that was actually sent, so the next Evaluate() can judge its result.
        /// </summary>
        public static void RecordCommand(double errX, double errY, double cmdX, double cmdY)
        {
            _pendingErrX = errX;
            _pendingErrY = errY;
            _pendingCmdX = cmdX;
            _pendingCmdY = cmdY;
            _hasPending = true;
        }

        /// <summary>
        /// Frame-rate compensation for the current update (computed once per update in Evaluate()).
        /// </summary>
        public static double StepScale() => _lastDtScale;

        /// <summary>
        /// Clears per-target state. The learned gain and tolerance are intentionally kept, because this
        /// is called on every frame that aim is idle and re-learning from scratch each time the user
        /// holds the aim key would reproduce the shake on every engagement.
        /// </summary>
        public static void Reset()
        {
            _pendingErrX = 0.0;
            _pendingErrY = 0.0;
            _pendingCmdX = 0.0;
            _pendingCmdY = 0.0;
            _hasPending = false;
            _cleanTicks = 0;
            _lastTickMs = -1;
            _lastDtScale = 1.0;
        }
    }
}
