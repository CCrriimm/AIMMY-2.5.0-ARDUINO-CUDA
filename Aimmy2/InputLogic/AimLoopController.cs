using Class;

namespace InputLogic
{
    /// <summary>
    /// Closed-loop gain governor for the aim controller.
    ///
    /// WHY THIS EXISTS
    /// ---------------
    /// Aim assist is a feedback loop:
    ///
    ///     capture frame -> detect target -> error = target - crosshair -> move mouse -> game camera moves -> repeat
    ///
    /// Every step of that loop has gain, and the total loop gain is
    ///
    ///     LoopGainPerTick   =  (1 - Sensitivity) * GamePixelsPerMouseCount
    ///     LoopGainPerSecond =  LoopGainPerTick * updatesPerSecond
    ///
    /// GamePixelsPerMouseCount is set by the game and grows steeply with optic magnification (a 4x
    /// scope swings the view several times further per mouse count than 1x does), and DirectX runs
    /// this loop far more times per second than GDI+ does, so the same settings that are stable at
    /// 2x / GDI+ become borderline unstable at 4x / DirectX.
    ///
    /// Past a loop gain of 1 the loop overshoots, then over-corrects, and settles into a limit cycle:
    /// the crosshair swings back and forth across the target while the camera visibly "shakes".
    /// That is exactly the symptom reported at 4x with DirectX capture.
    ///
    /// WHAT THIS CLASS DOES
    /// --------------------
    /// It measures the *quality* of the crosshair's recent motion instead of trusting the settings:
    ///
    ///   pathLength = how far the crosshair was commanded to travel (integral of |error|)
    ///   netTravel  = how far it actually got             (vector sum of error)
    ///
    /// For a real aim move, netTravel ~= pathLength (the crosshair travels a straight-ish line to the
    /// target). For a limit cycle the crosshair runs back and forth over the same spot, so pathLength
    /// keeps growing while netTravel cancels out to ~0.
    ///
    /// When that cancellation is detected the per-tick gain is reduced (fast, multiplicative) and, as
    /// soon as the motion becomes deliberate again, it is restored (slower, additive). The loop then
    /// self-balances at whatever gain is stable for the current zoom, sensitivity and capture rate,
    /// which removes the DirectX-vs-GDI+ difference instead of only masking it.
    ///
    /// It also normalises the step size by elapsed time so the same settings converge at the same
    /// *rate* regardless of how many frames per second the capture method delivers.
    /// </summary>
    internal static class AimLoopController
    {
        // Per-tick gain applied to the commanded movement. 1.0 = untouched behaviour.
        private const double MinGain = 0.16;
        private const double MaxGain = 1.0;

        // How aggressively the gain is cut while chattering, and how fast it is given back.
        // Recovery is deliberately gated: the gain only climbs again after this many consecutive
        // updates of clean, meaningful aim travel, so it cannot saw-tooth back up mid-oscillation.
        private const double GainCutPerTick = 0.12;
        private const double GainRecoverPerSecond = 3.0;
        private const int StableTicksBeforeRecovery = 6;

        // Path-efficiency band. cycleRatio = netTravel / pathLength.
        //   >= EfficientRatio (straight, deliberate aim)  -> full gain
        //   <= NoisyRatio     (pure back-and-forth shake) -> minimum gain
        private const double EfficientRatio = 0.50;
        private const double NoisyRatio = 0.15;

        // Errors below this are inside the lock-on deadzone and are ignored by the detector.
        private const double DeadZonePx = 3.0;

        // Direct chatter detection: a move that instantly reverses. This catches the dangerous limit
        // cycle in its first few updates instead of waiting for the rolling window to fill up.
        // The threshold scales with the current error so ordinary detection-box jitter (a few px on
        // top of a few px of error) cannot trip it - only a real back-and-forth of the aim can.
        private const double ReversalMinStepPx = 3.0;
        private const double ReversalMinStepRatio = 0.35;
        private const double ReversalPenaltyGain = 0.35;
        private const double ReversalPenaltyCap = 4.0;
        private const double ReversalTrigger = 3.0;

        // Time normalisation: step size is scaled towards this update period.
        private const double ReferenceDtMs = 16.0;
        private const double MinDtScale = 0.70;
        private const double MaxDtScale = 2.00;

        private static int _previousX = int.MinValue;
        private static int _previousY = int.MinValue;
        private static double _totalPath;
        private static double _netX;
        private static double _netY;
        private static double _previousStepX;
        private static double _previousStepY;
        private static bool _hasPreviousStep;
        private static double _reversalPenalty;
        private static bool _priming = true;
        private static int _primeTicks;
        private static int _stableTicks;
        private static double _gain = MaxGain;
        private static long _lastTickMs = -1;

        /// <summary>Last computed per-tick gain (0..1). Exposed for diagnostics/UI.</summary>
        public static double CurrentGain => _gain;

        /// <summary>Last computed path-efficiency ratio (netTravel / pathLength).</summary>
        public static double CurrentEfficiency { get; private set; } = 1.0;

        /// <summary>
        /// Feeds one detected-target sample (screen coordinates) into the governor and updates the gain.
        /// Call once per aim update, before the movement is computed.
        /// </summary>
        public static void Evaluate(int detectedX, int detectedY)
        {
            double halfW = WinAPICaller.ScreenWidth / 2.0;
            double halfH = WinAPICaller.ScreenHeight / 2.0;
            double errX = detectedX - halfW;
            double errY = detectedY - halfH;
            double magnitude = Math.Sqrt(errX * errX + errY * errY);

            long now = Environment.TickCount64;
            double dtMs = _lastTickMs < 0 ? ReferenceDtMs : Math.Clamp(now - _lastTickMs, 0.5, 100.0);
            _lastTickMs = now;

            // ── Prime after (re)acquisition ──
            // The first few ticks after the crosshair snaps onto a target contain a huge, deliberate
            // move. Measuring it would look like noise, so ignore it and start clean.
            if (_priming || _previousX == int.MinValue)
            {
                if (++_primeTicks >= 3)
                {
                    _priming = false;
                }
                _previousX = detectedX;
                _previousY = detectedY;
                _totalPath = 0;
                _netX = 0;
                _netY = 0;
                _stableTicks = StableTicksBeforeRecovery;
                RecoverGain(dtMs);
                return;
            }

            if (magnitude < DeadZonePx)
            {
                // Locked on: no command is being issued, so nothing can oscillate. Let the gain recover.
                _totalPath = 0;
                _netX = 0;
                _netY = 0;
                _previousX = detectedX;
                _previousY = detectedY;
                _hasPreviousStep = false;
                _reversalPenalty = Math.Max(0.0, _reversalPenalty - 0.5);
                _stableTicks = StableTicksBeforeRecovery;
                RecoverGain(dtMs);
                return;
            }

            double stepX = errX - (_previousX - halfW);
            double stepY = errY - (_previousY - halfH);
            double step = Math.Sqrt(stepX * stepX + stepY * stepY);

            // ── Direct chatter detection ──
            // A move that immediately reverses, repeatedly, cannot be aim: it is the loop fighting
            // itself. The step must also be significant compared to the current error, otherwise
            // ordinary bounding-box noise would be mistaken for oscillation.
            if (_hasPreviousStep)
            {
                double dot = stepX * _previousStepX + stepY * _previousStepY;
                double reversalThreshold = Math.Max(ReversalMinStepPx, ReversalMinStepRatio * magnitude);
                if (dot < 0 && step > reversalThreshold)
                {
                    _reversalPenalty = Math.Min(ReversalPenaltyCap, _reversalPenalty + 1.0);
                }
                else
                {
                    _reversalPenalty = Math.Max(0.0, _reversalPenalty - 0.25);
                }
            }

            _previousStepX = stepX;
            _previousStepY = stepY;
            _hasPreviousStep = true;

            if (_reversalPenalty >= ReversalTrigger)
            {
                double rateScale = Math.Clamp(dtMs / ReferenceDtMs, 0.25, 4.0);
                _gain = Math.Max(MinGain, _gain - ReversalPenaltyGain * rateScale);
            }

            // Rolling window (kept in "error units", which are screen pixels).
            _totalPath += step;
            _netX += stepX;
            _netY += stepY;

            // Decay so the window tracks recent behaviour (~0.25 s at 60 Hz).
            double decay = Math.Exp(-dtMs / 250.0);
            _totalPath *= decay;
            _netX *= decay;
            _netY *= decay;

            _previousX = detectedX;
            _previousY = detectedY;

            double net = Math.Sqrt(_netX * _netX + _netY * _netY);

            // Only judge efficiency once enough motion has accumulated to be meaningful.
            if ((_totalPath < magnitude * 2.0 || _totalPath < 20.0) && _reversalPenalty < 2.0)
            {
                CurrentEfficiency = 1.0;
                _stableTicks = StableTicksBeforeRecovery;
                RecoverGain(dtMs);
                return;
            }

            double efficiency = Math.Clamp(net / _totalPath, 0.0, 1.0);
            CurrentEfficiency = efficiency;

            if (efficiency >= EfficientRatio && _reversalPenalty < 1.0)
            {
                // Deliberate travel: this update counts towards giving the gain back.
                _stableTicks++;
                RecoverGain(dtMs);
            }
            else if (efficiency <= NoisyRatio)
            {
                // Chattering in place: cut the gain, proportionally to the update rate so the
                // suppression rate is the same on DirectX (many updates/s) and GDI+ (fewer).
                _stableTicks = 0;
                double rateScale = Math.Clamp(dtMs / ReferenceDtMs, 0.25, 4.0);
                _gain = Math.Max(MinGain, _gain - GainCutPerTick * rateScale);
            }
            else
            {
                // Transition band: bias towards stability without slamming the gain.
                _stableTicks = 0;
                double t = Math.Clamp((efficiency - NoisyRatio) / (EfficientRatio - NoisyRatio), 0.0, 1.0);
                double target = MinGain + t * (MaxGain - MinGain);
                _gain += (target - _gain) * 0.25;
                _gain = Math.Clamp(_gain, MinGain, MaxGain);
            }
        }

        /// <summary>
        /// Gives gain back only after a run of clean updates, so a limit cycle cannot saw-tooth
        /// between "suppress" and "release" while it is still oscillating.
        /// </summary>
        private static void RecoverGain(double dtMs)
        {
            if (_stableTicks < StableTicksBeforeRecovery) return;
            _gain = Math.Min(MaxGain, _gain + GainRecoverPerSecond * (dtMs / 1000.0));
        }

        /// <summary>
        /// Frame-rate compensation: returns the multiplier that turns a fixed per-update step into a
        /// step that depends on how long the previous update took, so the loop converges at the same
        /// speed whether the capture method delivers 60 or 400 updates per second.
        /// </summary>
        public static double StepScale()
        {
            long now = Environment.TickCount64;
            if (_lastTickMs < 0) return 1.0;
            double dtMs = Math.Clamp(now - _lastTickMs, 1.0, 60.0);
            return Math.Clamp(dtMs / ReferenceDtMs, MinDtScale, MaxDtScale);
        }

        public static void Reset()
        {
            _previousX = int.MinValue;
            _previousY = int.MinValue;
            _totalPath = 0;
            _netX = 0;
            _netY = 0;
            _previousStepX = 0;
            _previousStepY = 0;
            _hasPreviousStep = false;
            _reversalPenalty = 0;
            _priming = true;
            _primeTicks = 0;
            _stableTicks = 0;
            _gain = MaxGain;
            _lastTickMs = -1;
            CurrentEfficiency = 1.0;
        }
    }
}
