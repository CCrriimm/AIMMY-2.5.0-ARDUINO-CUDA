using AILogic;
using Aimmy2.Class;
using Aimmy2.Other;
using Class;
using InputLogic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json.Linq;
using Other;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows;
using Visuality;
using static AILogic.MathUtil;
using static Other.LogManager;

namespace Aimmy2.AILogic
{
    internal class AIManager : IDisposable
    {
        #region Variables

        private int _currentImageSize;
        private readonly object _sizeLock = new object();
        private volatile bool _sizeChangePending = false;

        public void RequestSizeChange(int newSize)
        {
            lock (_sizeLock)
            {
                _sizeChangePending = true;
            }
        }

        // Dynamic properties instead of constants
        public int IMAGE_SIZE => _currentImageSize;
        private int NUM_DETECTIONS { get; set; } = 8400; // Will be set dynamically for dynamic models
        private bool IsDynamicModel { get; set; } = false;

        // Public static property to check if current loaded model is dynamic
        public static bool CurrentModelIsDynamic { get; private set; } = false;
        private int ModelFixedSize { get; set; } = 640; // Store the fixed size for non-dynamic models
        private int NUM_CLASSES { get; set; } = 1;
        private Dictionary<int, string> _modelClasses = new Dictionary<int, string>
        {
            { 0, "enemy" }
        };
        public Dictionary<int, string> ModelClasses => _modelClasses; // apparently this is better than making _modelClasses public
        public static event Action<Dictionary<int, string>>? ClassesUpdated;
        public static event Action<int>? ImageSizeUpdated;
        public static event Action<bool>? DynamicModelStatusChanged;

        private const int SAVE_FRAME_COOLDOWN_MS = 500;

        private DateTime lastSavedTime = DateTime.MinValue;
        private List<string>? _outputNames;
        private RectangleF LastDetectionBox;
        private KalmanPrediction kalmanPrediction;
        private WiseTheFoxPrediction wtfpredictionManager;

        private byte[]? _bitmapBuffer; // Reusable buffer for bitmap operations

        // Display-aware properties
        private int ScreenWidth => DisplayManager.ScreenWidth;
        private int ScreenHeight => DisplayManager.ScreenHeight;
        private int ScreenLeft => DisplayManager.ScreenLeft;
        private int ScreenTop => DisplayManager.ScreenTop;

        private readonly RunOptions? _modeloptions;
        private InferenceSession? _onnxModel;

        private Thread? _aiLoopThread;
        private volatile bool _isAiLoopRunning;

        // For Auto-Labelling Data System
        private bool PlayerFound = false;

        // Sticky-Aim
        private Prediction? _currentTarget = null;
        private int _consecutiveFramesWithoutTarget = 0;
        private const int MAX_FRAMES_WITHOUT_TARGET = 3; // Allow 3 frames of target loss

        // Enhanced Sticky Aim State
        private float _lastTargetVelocityX = 0f;
        private float _lastTargetVelocityY = 0f;

        // Aim rate limiter - prevents sending mouse commands faster than the display can refresh
        // DirectX can capture at 200+ FPS but games render at 60-144 FPS.
        // Sending aim commands faster than the game renders causes overshoot oscillation (shaking).
        private readonly Stopwatch _aimRateTimer = Stopwatch.StartNew();
        private const long AIM_MIN_INTERVAL_MS = 2; // ~500Hz max aim rate (cleanly passes 144Hz-240Hz monitor frames)

        private double CenterXTranslated = 0;
        private double CenterYTranslated = 0;

        // Benchmarking
        private int iterationCount = 0;
        private long totalTime = 0;

        private int detectedX { get; set; }
        private int detectedY { get; set; }

        public double AIConf = 0;
        private static int targetX, targetY;

        // Pre-calculated values - now dynamic
        private float _scaleX => ScreenWidth / (float)IMAGE_SIZE;
        private float _scaleY => ScreenHeight / (float)IMAGE_SIZE;

        // Tensor reuse (model inference)
        private DenseTensor<float>? _reusableTensor;
        private float[]? _reusableInputArray;
        private List<NamedOnnxValue>? _reusableInputs;
        private string _inputName = "images";

        private enum ModelOutputFormat
        {
            YOLO_Raw_ChannelsFirst,      // [1, 4 + NUM_CLASSES, NUM_DETECTIONS] (YOLOv8/v9/v11/v26 raw)
            YOLO_Raw_ChannelsLast,       // [1, NUM_DETECTIONS, 4 + NUM_CLASSES] (Transposed raw)
            YOLO_EndToEnd_BoxesFirst,    // [1, NUM_DETECTIONS, 6] (YOLOv26 / YOLOv10 end2end: x1, y1, x2, y2, score, class)
            YOLO_EndToEnd_ChannelsFirst  // [1, 6, NUM_DETECTIONS]
        }
        private ModelOutputFormat _outputFormat = ModelOutputFormat.YOLO_Raw_ChannelsFirst;

        // Benchmarking
        private readonly Dictionary<string, BenchmarkData> _benchmarks = new();
        private readonly object _benchmarkLock = new();


        private readonly CaptureManager _captureManager = new();
        #endregion Variables

        #region Benchmarking

        private class BenchmarkData
        {
            public long TotalTime { get; set; }
            public int CallCount { get; set; }
            public long MinTime { get; set; } = long.MaxValue;
            public long MaxTime { get; set; }
            public double AverageTime => CallCount > 0 ? (double)TotalTime / CallCount : 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private IDisposable Benchmark(string name)
        {
            return new BenchmarkScope(this, name);
        }

        private class BenchmarkScope : IDisposable
        {
            private readonly AIManager _manager;
            private readonly string _name;
            private readonly Stopwatch _sw;

            public BenchmarkScope(AIManager manager, string name)
            {
                _manager = manager;
                _name = name;
                _sw = Stopwatch.StartNew();
            }

            public void Dispose()
            {
                _sw.Stop();
                _manager.RecordBenchmark(_name, _sw.ElapsedMilliseconds);
            }
        }

        private void RecordBenchmark(string name, long elapsedMs)
        {
            lock (_benchmarkLock)
            {
                if (!_benchmarks.TryGetValue(name, out var data))
                {
                    data = new BenchmarkData();
                    _benchmarks[name] = data;
                }

                data.TotalTime += elapsedMs;
                data.CallCount++;
                data.MinTime = Math.Min(data.MinTime, elapsedMs);
                data.MaxTime = Math.Max(data.MaxTime, elapsedMs);
            }
        }

        public void PrintBenchmarks()
        {
            lock (_benchmarkLock)
            {
                var lines = new List<string>
                {
                    "=== AIManager Performance Benchmarks ==="
                };

                foreach (var kvp in _benchmarks.OrderBy(x => x.Key))
                {
                    var data = kvp.Value;
                    lines.Add($"{kvp.Key}: Avg={data.AverageTime:F2}ms, Min={data.MinTime}ms, Max={data.MaxTime}ms, Count={data.CallCount}");
                }

                lines.Add($"Overall FPS: {(iterationCount > 0 ? 1000.0 / (totalTime / (double)iterationCount) : 0):F2}");

                //File.WriteAllLines("AIManager_Benchmarks.txt", lines);

                Log(LogLevel.Info, string.Join(Environment.NewLine, lines));
            }
        }

        #endregion Benchmarking

        public AIManager(string modelPath)
        {
            // Initialize the cached image size
            _currentImageSize = int.Parse(Dictionary.dropdownState["Image Size"]);

            // Initialize DXGI capture for current display
            if (Dictionary.dropdownState["Screen Capture Method"] == "DirectX")
            {
                _captureManager.InitializeDxgiDuplication();
            }

            kalmanPrediction = new KalmanPrediction();
            wtfpredictionManager = new WiseTheFoxPrediction();

            _modeloptions = new RunOptions();

            var sessionOptions = new SessionOptions
            {
                EnableCpuMemArena = true,
                EnableMemoryPattern = false,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                InterOpNumThreads = 1,
                IntraOpNumThreads = 4
            };

            // Attempt to load via DirectML (else fallback to CPU)
            Task.Run(() => InitializeModel(sessionOptions, modelPath));
        }

        #region Models

        private async Task InitializeModel(SessionOptions sessionOptions, string modelPath)
        {
            using (Benchmark("ModelInitialization"))
            {
                try
                {
                    await LoadModelAsync(modelPath);
                }
                catch (Exception ex)
                {
                    Log(LogLevel.Error, $"Error starting the model via DirectML: {ex.Message}\n\nFalling back to CPU, performance may be poor.", true);

                    try
                    {
                        await LoadModelAsync(modelPath);
                    }
                    catch (Exception e)
                    {
                        Log(LogLevel.Error, $"Error starting the model via CPU: {e.Message}, you won't be able to aim assist at all.", true);
                    }
                }

                FileManager.CurrentlyLoadingModel = false;
            }
        }

        private Task LoadModelAsync(string modelPath)
        {
            // Ensure performance optimizations and GPU scheduling priorities are applied
            bool enableRealtime = Dictionary.toggleState.TryGetValue("Realtime GPU Priority", out var rgp) ? Convert.ToBoolean(rgp) : true;
            PerformanceOptimizer.OptimizeAll(enableRealtime);

            string activeProvider = "CPU";
            InferenceSession? session = null;

            string preferredProvider = Dictionary.dropdownState.TryGetValue("Execution Provider", out var ep) && ep != null
                ? ep.ToString()!
                : "";

            // If TensorRT is chosen, show warning notice about load time
            if (preferredProvider.Equals("TensorRT", StringComparison.OrdinalIgnoreCase))
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    new NoticeBar("Loading model with TensorRT, expect long model load time.", 5000).Show();
                });
            }

            string cacheDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "bin", "tensorrt_cache");
            try
            {
                Directory.CreateDirectory(cacheDir);
            }
            catch { }

            SessionOptions CreateBaseOptions()
            {
                return new SessionOptions
                {
                    EnableCpuMemArena = true,
                    EnableMemoryPattern = false,
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                    InterOpNumThreads = 1,
                    IntraOpNumThreads = 4
                };
            }

            bool TryTensorRT()
            {
                try
                {
                    var trtOptions = CreateBaseOptions();
                    try
                    {
                        var providerOptions = new OrtTensorRTProviderOptions();
                        providerOptions.UpdateOptions(new Dictionary<string, string>
                        {
                            { "device_id", "0" },
                            { "trt_engine_cache_enable", "1" },
                            { "trt_engine_cache_path", cacheDir },
                            { "trt_fp16_enable", "1" }
                        });
                        trtOptions.AppendExecutionProvider_Tensorrt(providerOptions);
                    }
                    catch
                    {
                        trtOptions.AppendExecutionProvider_Tensorrt(0);
                    }

                    try { trtOptions.AppendExecutionProvider_CUDA(0); } catch { }

                    session = new InferenceSession(modelPath, trtOptions);
                    activeProvider = "TensorRT";
                    Log(LogLevel.Info, "Successfully initialized ONNX session with TensorRT execution provider.");
                    return true;
                }
                catch (Exception ex)
                {
                    Log(LogLevel.Warning, $"TensorRT provider failed ({ex.Message}).");
                    session?.Dispose();
                    session = null;
                    return false;
                }
            }

            bool TryCUDA()
            {
                try
                {
                    var cudaOptions = CreateBaseOptions();
                    cudaOptions.AppendExecutionProvider_CUDA(0);
                    session = new InferenceSession(modelPath, cudaOptions);
                    activeProvider = "CUDA";
                    Log(LogLevel.Info, "Successfully initialized ONNX session with CUDA execution provider.");
                    return true;
                }
                catch (Exception ex)
                {
                    Log(LogLevel.Warning, $"CUDA provider unavailable ({ex.Message}).");
                    session?.Dispose();
                    session = null;
                    return false;
                }
            }

            bool TryDML()
            {
                try
                {
                    var dmlOptions = CreateBaseOptions();
                    dmlOptions.AppendExecutionProvider_DML(0);
                    session = new InferenceSession(modelPath, dmlOptions);
                    activeProvider = "DML";
                    Log(LogLevel.Info, "Successfully initialized ONNX session with DirectML execution provider.");
                    return true;
                }
                catch (Exception ex)
                {
                    Log(LogLevel.Warning, $"DirectML provider unavailable ({ex.Message}).");
                    session?.Dispose();
                    session = null;
                    return false;
                }
            }

            bool TryCPU()
            {
                try
                {
                    var cpuOptions = CreateBaseOptions();
                    cpuOptions.AppendExecutionProvider_CPU();
                    session = new InferenceSession(modelPath, cpuOptions);
                    activeProvider = "CPU";
                    Log(LogLevel.Warning, "Initialized ONNX session with CPU execution provider (performance will be reduced).");
                    return true;
                }
                catch (Exception ex)
                {
                    Log(LogLevel.Error, $"Error loading model via CPU: {ex.Message}", true);
                    session?.Dispose();
                    session = null;
                    return false;
                }
            }

            if (preferredProvider.Equals("TensorRT", StringComparison.OrdinalIgnoreCase))
            {
                if (!TryTensorRT())
                {
                    Log(LogLevel.Warning, "TensorRT failed; falling back to CUDA...");
                    if (!TryCUDA() && !TryDML())
                    {
                        TryCPU();
                    }
                }
            }
            else if (preferredProvider.Equals("CUDA", StringComparison.OrdinalIgnoreCase))
            {
                if (!TryCUDA())
                {
                    Log(LogLevel.Warning, "CUDA failed; falling back to DirectML...");
                    if (!TryDML())
                    {
                        TryCPU();
                    }
                }
            }
            else if (preferredProvider.Equals("DirectML", StringComparison.OrdinalIgnoreCase))
            {
                if (!TryDML())
                {
                    TryCPU();
                }
            }
            else if (preferredProvider.Equals("CPU", StringComparison.OrdinalIgnoreCase))
            {
                TryCPU();
            }
            else
            {
                // Blank / unselected provider on first run: try CUDA -> DirectML -> CPU
                if (!TryCUDA())
                {
                    if (!TryDML())
                    {
                        TryCPU();
                    }
                }
            }

            if (session == null)
            {
                Log(LogLevel.Error, "Failed to initialize any ONNX execution provider.", true);
                return Task.CompletedTask;
            }

            _onnxModel = session;
            _outputNames = new List<string>(_onnxModel.OutputMetadata.Keys);

            Application.Current.Dispatcher.Invoke(() =>
            {
                new NoticeBar($"USING {activeProvider}", 2500).Show();
            });

            // Validate the onnx model output shape (ensure model is OnnxV8)
            if (!ValidateOnnxShape())
            {
                _onnxModel?.Dispose();
                return Task.CompletedTask;
            }

            // Pre-allocate bitmap buffer
            _bitmapBuffer = new byte[3 * IMAGE_SIZE * IMAGE_SIZE];

            // Begin the loop
            _isAiLoopRunning = true;
            _aiLoopThread = new Thread(AiLoop)
            {
                IsBackground = true,
                Priority = ThreadPriority.Highest // Top priority for AI thread to avoid frame starvation
            };
            _aiLoopThread.Start();
            return Task.CompletedTask;
        }

        private bool ValidateOnnxShape()
        {
            if (_onnxModel != null)
            {
                var inputMetadata = _onnxModel.InputMetadata;
                var outputMetadata = _onnxModel.OutputMetadata;

                Log(LogLevel.Info, "=== Model Metadata ===");
                Log(LogLevel.Info, "Input Metadata:");

                bool isDynamic = false;
                int fixedInputSize = 0;

                _inputName = inputMetadata.Keys.FirstOrDefault() ?? "images";

                foreach (var kvp in inputMetadata)
                {
                    string dimensionsStr = string.Join("x", kvp.Value.Dimensions);
                    Log(LogLevel.Info, $"  Name: {kvp.Key}, Dimensions: {dimensionsStr}");

                    // Check if model is dynamic (dimensions are -1)
                    if (kvp.Value.Dimensions.Any(d => d == -1))
                    {
                        isDynamic = true;
                    }
                    else if (kvp.Value.Dimensions.Length == 4)
                    {
                        // For fixed models, check if it's the expected format (1x3xHxW)
                        fixedInputSize = kvp.Value.Dimensions[2]; // Height should equal Width for square models
                    }
                }

                Log(LogLevel.Info, "Output Metadata:");
                foreach (var kvp in outputMetadata)
                {
                    string dimensionsStr = string.Join("x", kvp.Value.Dimensions);
                    Log(LogLevel.Info, $"  Name: {kvp.Key}, Dimensions: {dimensionsStr}");
                }

                IsDynamicModel = isDynamic;
                CurrentModelIsDynamic = isDynamic;

                LoadClasses();

                // Determine Output Format from primary output metadata
                var primaryOutput = outputMetadata.Values.FirstOrDefault();
                int[] dims = primaryOutput?.Dimensions ?? Array.Empty<int>();

                if (dims.Length == 3)
                {
                    if (dims[2] == 6)
                    {
                        // YOLOv26 / YOLOv10 End-to-End: [1, N, 6] (x1, y1, x2, y2, score, class)
                        _outputFormat = ModelOutputFormat.YOLO_EndToEnd_BoxesFirst;
                        NUM_DETECTIONS = dims[1] > 0 ? dims[1] : 300;
                        Log(LogLevel.Info, $"[Model Detection] Detected YOLOv26/YOLOv10 End-to-End format with {NUM_DETECTIONS} detections.", true, 3000);
                    }
                    else if (dims[1] == 6 && dims[2] > 6)
                    {
                        _outputFormat = ModelOutputFormat.YOLO_EndToEnd_ChannelsFirst;
                        NUM_DETECTIONS = dims[2];
                        Log(LogLevel.Info, $"[Model Detection] Detected YOLO End-to-End (ChannelsFirst) format with {NUM_DETECTIONS} detections.", true, 3000);
                    }
                    else if (dims[1] > 0 && dims[2] > 0 && dims[1] < dims[2])
                    {
                        _outputFormat = ModelOutputFormat.YOLO_Raw_ChannelsFirst;
                        NUM_DETECTIONS = dims[2];
                    }
                    else if (dims[1] > 0 && dims[2] > 0 && dims[1] >= dims[2])
                    {
                        _outputFormat = ModelOutputFormat.YOLO_Raw_ChannelsLast;
                        NUM_DETECTIONS = dims[1];
                    }
                    else
                    {
                        _outputFormat = ModelOutputFormat.YOLO_Raw_ChannelsFirst;
                    }
                }

                if (IsDynamicModel)
                {
                    if (_outputFormat != ModelOutputFormat.YOLO_EndToEnd_BoxesFirst && _outputFormat != ModelOutputFormat.YOLO_EndToEnd_ChannelsFirst)
                    {
                        NUM_DETECTIONS = CalculateNumDetections(IMAGE_SIZE);
                    }
                    ImageSizeUpdated?.Invoke(IMAGE_SIZE);
                    Log(LogLevel.Info, $"Loaded dynamic model - using selected image size {IMAGE_SIZE}x{IMAGE_SIZE} with {NUM_DETECTIONS} detections", true, 3000);
                }
                else
                {
                    // For fixed models, auto-adjust image size if needed
                    ModelFixedSize = fixedInputSize;

                    // List of supported sizes
                    var supportedSizes = new[] { "640", "512", "416", "320", "256", "160" };
                    var fixedSizeStr = fixedInputSize.ToString();

                    if (!supportedSizes.Contains(fixedSizeStr))
                    {
                        Log(LogLevel.Error,
                            $"Model requires unsupported size {fixedInputSize}x{fixedInputSize}. Supported sizes are: {string.Join(", ", supportedSizes)}",
                            true, 10000);
                        return false;
                    }

                    if (_outputFormat != ModelOutputFormat.YOLO_EndToEnd_BoxesFirst && _outputFormat != ModelOutputFormat.YOLO_EndToEnd_ChannelsFirst)
                    {
                        NUM_DETECTIONS = CalculateNumDetections(fixedInputSize);
                    }
                    _currentImageSize = fixedInputSize;

                    if (fixedInputSize != int.Parse(Dictionary.dropdownState["Image Size"]))
                    {
                        // Auto-adjust the image size to match the model
                        Log(LogLevel.Warning,
                            $"Fixed-size model expects {fixedInputSize}x{fixedInputSize}. Automatically adjusting Image Size setting.",
                            true, 3000);

                        Dictionary.dropdownState["Image Size"] = fixedSizeStr;

                        // Update the UI dropdown if it exists
                        Application.Current?.Dispatcher.BeginInvoke(() =>
                        {
                            try
                            {
                                var mainWindow = Application.Current.Windows.OfType<MainWindow>().FirstOrDefault();
                                if (mainWindow?.SettingsMenuControlInstance != null)
                                {
                                    mainWindow.SettingsMenuControlInstance.UpdateImageSizeDropdown(fixedSizeStr);
                                }
                            }
                            catch { }
                        });
                    }

                    ImageSizeUpdated?.Invoke(fixedInputSize);

                    // Ensure rank 3 tensor
                    if (dims.Length != 3)
                    {
                        Log(LogLevel.Error,
                            $"Output tensor has unexpected rank {dims.Length}. Expected a 3D tensor (e.g. [1, 300, 6] or [1, {4 + NUM_CLASSES}, {NUM_DETECTIONS}]).",
                            true, 10000);
                        return false;
                    }

                    Log(LogLevel.Info, $"Loaded fixed-size model: {fixedInputSize}x{fixedInputSize} ({_outputFormat})", true, 2000);
                }

                // Notify UI about dynamic model status
                DynamicModelStatusChanged?.Invoke(IsDynamicModel);

                return true;
            }

            return false;
        }

        private void LoadClasses()
        {
            if (_onnxModel == null) return;
            _modelClasses.Clear();

            try
            {
                var metadata = _onnxModel.ModelMetadata;

                if (metadata != null &&
                    metadata.CustomMetadataMap.TryGetValue("names", out string? value) &&
                    !string.IsNullOrEmpty(value))
                {
                    JObject data = JObject.Parse(value);
                    if (data != null && data.Type == JTokenType.Object)
                    {
                        //int maxClassId = -1;
                        foreach (var item in data)
                        {
                            if (int.TryParse(item.Key, out int classId) && item.Value.Type == JTokenType.String)
                            {
                                _modelClasses[classId] = item.Value.ToString();
                            }
                        }
                        NUM_CLASSES = _modelClasses.Count > 0 ? _modelClasses.Keys.Max() + 1 : 1;
                        Log(LogLevel.Info, $"Loaded {_modelClasses.Count} class(es) from model metadata: {data.ToString(Newtonsoft.Json.Formatting.None)}", false);
                    }
                    else
                    {
                        Log(LogLevel.Error, "Model metadata 'names' field is not a valid JSON object.", true);
                    }
                }
                else
                {
                    Log(LogLevel.Error, "Model metadata does not contain 'names' field for classes.", true);
                }
                ClassesUpdated?.Invoke(new Dictionary<int, string>(_modelClasses));
            }
            catch (Exception ex)
            {
                Log(LogLevel.Error, $"Error loading classes: {ex.Message}", true);
            }
        }

        #endregion Models

        #region AI

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool ShouldPredict() =>
            Dictionary.toggleState["Show Detected Player"] ||
            Dictionary.toggleState["Constant AI Tracking"] ||
            InputBindingManager.IsHoldingBinding("Aim Keybind") ||
            InputBindingManager.IsHoldingBinding("Second Aim Keybind") ||
            (Dictionary.toggleState["Insta-Flick"] && (InputBindingManager.IsHoldingBinding("Insta-Flick Keybind") || InputBindingManager.IsHoldingBinding("Second Insta-Flick Keybind")));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool ShouldProcess() =>
            Dictionary.toggleState["Aim Assist"] ||
            Dictionary.toggleState["Show Detected Player"] ||
            Dictionary.toggleState["Auto Trigger"] ||
            Dictionary.toggleState["Insta-Flick"];

        private async void AiLoop()
        {
            Stopwatch stopwatch = new();
            DetectedPlayerWindow? DetectedPlayerOverlay = Dictionary.DetectedPlayerOverlay;

            while (_isAiLoopRunning)
            {
                // Check for pending size changes at the start of each iteration
                bool shouldPause = false;
                lock (_sizeLock)
                {
                    if (_sizeChangePending)
                    {
                        shouldPause = true;
                    }
                }

                if (shouldPause)
                {
                    // Skip this iteration to allow clean shutdown
                    await Task.Delay(10);
                    continue;
                }

                stopwatch.Restart();

                // Handle any pending display changes
                _captureManager.HandlePendingDisplayChanges();

                using (Benchmark("AILoopIteration"))
                {
                    UpdateFOV();

                    if (ShouldProcess())
                    {
                        if (ShouldPredict())
                        {
                            Prediction? closestPrediction;
                            bool frameAcquired;
                            using (Benchmark("GetClosestPrediction"))
                            {
                                (closestPrediction, frameAcquired) = await GetClosestPrediction();
                            }

                            if (!frameAcquired)
                            {
                                await Task.Delay(1);
                                continue;
                            }

                            if (closestPrediction == null)
                            {
                                DisableOverlay(DetectedPlayerOverlay!);
                                continue;
                            }

                            using (Benchmark("AutoTrigger"))
                            {
                                await AutoTrigger(closestPrediction);
                            }

                            using (Benchmark("CalculateCoordinates"))
                            {
                                CalculateCoordinates(DetectedPlayerOverlay, closestPrediction, _scaleX, _scaleY);
                            }

                            using (Benchmark("HandleAim"))
                            {
                                HandleAim(closestPrediction);
                            }

                            totalTime += stopwatch.ElapsedMilliseconds;
                            iterationCount++;
                        }
                        else
                        {
                            // Processing so we are at the ready but not holding right/click.
                            MouseManager.ResetMovement();
                            await Task.Delay(1);
                        }
                    }
                    else
                    {
                        // No work to do—sleep briefly to free up CPU
                        MouseManager.ResetMovement();
                        await Task.Delay(1);
                    }
                }

                stopwatch.Stop();
            }
        }

        #region AI Loop Functions

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private async Task AutoTrigger(Prediction? closestPrediction = null)
        {
            // if auto trigger is disabled,
            // or if the aim keybinds are not held,
            // or if constant AI tracking is enabled,
            // we check for spray release and return
            if (!Dictionary.toggleState["Auto Trigger"] ||
                !(InputBindingManager.IsHoldingBinding("Aim Keybind") && !InputBindingManager.IsHoldingBinding("Second Aim Keybind")) ||
                Dictionary.toggleState["Constant AI Tracking"]) // this logic is a bit weird, but it works.
                                                                // but it might need to be revised
            {
                CheckSprayRelease();
                return;
            }

            if (Dictionary.toggleState["FOV Action"] && closestPrediction != null)
            {
                float actionSize = (float)Dictionary.sliderSettings["FOV Action Size"];
                float center = IMAGE_SIZE / 2.0f;
                string fovStyle = Dictionary.dropdownState.TryGetValue("FOV Style", out var s) ? s.ToString() : "Circle";
                if (!IsInsideFovAction(closestPrediction, center, actionSize / 2.0f, fovStyle))
                {
                    CheckSprayRelease();
                    return;
                }
            }


            if (Dictionary.toggleState["Spray Mode"])
            {
                await MouseManager.DoTriggerClick(LastDetectionBox);
                return;
            }


            if (Dictionary.toggleState["Cursor Check"])
            {
                var mousePos = WinAPICaller.GetCursorPosition();

                if (!DisplayManager.IsPointInCurrentDisplay(new System.Windows.Point(mousePos.X, mousePos.Y)))
                {
                    return;
                }

                if (LastDetectionBox.Contains(mousePos.X, mousePos.Y))
                {
                    await MouseManager.DoTriggerClick(LastDetectionBox);
                }
            }
            else
            {
                await MouseManager.DoTriggerClick();
            }

            if (!Dictionary.toggleState["Aim Assist"] || !Dictionary.toggleState["Show Detected Player"]) return;

        }
        private void CheckSprayRelease()
        {
            if (!Dictionary.toggleState["Spray Mode"]) return;

            // if auto trigger is disabled, we reset the spray state
            // if the aim keybinds are not held, we reset the spray state
            bool shouldSpray = Dictionary.toggleState["Auto Trigger"] &&
                (InputBindingManager.IsHoldingBinding("Aim Keybind") && InputBindingManager.IsHoldingBinding("Second Aim Keybind")); //||
                                                                                                                                     //Dictionary.toggleState["Constant AI Tracking"];

            // spray mode might need to be revised - taylor
            if (!shouldSpray)
            {
                MouseManager.ResetSprayState();
            }
        }

        private async void UpdateFOV()
        {
            if (Dictionary.dropdownState["Detection Area Type"] == "Closest to Mouse" && Dictionary.toggleState["FOV"])
            {
                var mousePosition = WinAPICaller.GetCursorPosition();

                // Check if mouse is on the current display
                if (!DisplayManager.IsPointInCurrentDisplay(new System.Windows.Point(mousePosition.X, mousePosition.Y)))
                {
                    // Mouse is on a different display - don't update FOV position
                    return;
                }

                // Translate mouse position relative to current display
                var displayRelativeX = mousePosition.X - DisplayManager.ScreenLeft;
                var displayRelativeY = mousePosition.Y - DisplayManager.ScreenTop;

                await Application.Current.Dispatcher.BeginInvoke(() =>
                    Dictionary.FOVWindow.FOVStrictEnclosure.Margin = new Thickness(
                        Convert.ToInt16(displayRelativeX / WinAPICaller.scalingFactorX) - 320, // this is based off the window size, not the size of the model -whip
                        Convert.ToInt16(displayRelativeY / WinAPICaller.scalingFactorY) - 320, 0, 0));
            }
        }

        private static void DisableOverlay(DetectedPlayerWindow DetectedPlayerOverlay)
        {
            if (Dictionary.toggleState["Show Detected Player"] && Dictionary.DetectedPlayerOverlay != null)
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    if (Dictionary.toggleState["Show AI Confidence"])
                    {
                        DetectedPlayerOverlay!.DetectedPlayerConfidence.Opacity = 0;
                    }

                    if (Dictionary.toggleState["Show Tracers"])
                    {
                        DetectedPlayerOverlay!.DetectedTracers.Opacity = 0;
                    }

                    DetectedPlayerOverlay!.DetectedPlayerFocus.Opacity = 0;
                });
            }
        }

        private void UpdateOverlay(DetectedPlayerWindow DetectedPlayerOverlay, Prediction closestPrediction)
        {
            var scalingFactorX = WinAPICaller.scalingFactorX;
            var scalingFactorY = WinAPICaller.scalingFactorY;

            // Convert screen coordinates to display-relative coordinates
            var displayRelativeX = LastDetectionBox.X - DisplayManager.ScreenLeft;
            var displayRelativeY = LastDetectionBox.Y - DisplayManager.ScreenTop;

            // Calculate center position in display-relative coordinates
            var centerX = Convert.ToInt16(displayRelativeX / scalingFactorX) + (LastDetectionBox.Width / 2.0);
            var centerY = Convert.ToInt16(displayRelativeY / scalingFactorY);

            Application.Current.Dispatcher.Invoke(() =>
            {
                if (Dictionary.toggleState["Show AI Confidence"])
                {
                    DetectedPlayerOverlay.DetectedPlayerConfidence.Opacity = 1;
                    DetectedPlayerOverlay.DetectedPlayerConfidence.Content = $"{closestPrediction.ClassName}: {Math.Round((AIConf * 100), 2)}%";

                    var labelEstimatedHalfWidth = DetectedPlayerOverlay.DetectedPlayerConfidence.ActualWidth / 2.0;
                    DetectedPlayerOverlay.DetectedPlayerConfidence.Margin = new Thickness(
                        centerX - labelEstimatedHalfWidth,
                        centerY - DetectedPlayerOverlay.DetectedPlayerConfidence.ActualHeight - 2, 0, 0);
                }
                var showTracers = Dictionary.toggleState["Show Tracers"];
                DetectedPlayerOverlay.DetectedTracers.Opacity = showTracers ? 1 : 0;
                if (showTracers)
                {
                    var tracerPosition = Dictionary.dropdownState["Tracer Position"];

                    var boxTop = centerY;
                    var boxBottom = centerY + LastDetectionBox.Height;
                    var boxHorizontalCenter = centerX;
                    var boxVerticalCenter = centerY + (LastDetectionBox.Height / 2.0);
                    var boxLeft = centerX - (LastDetectionBox.Width / 2.0);
                    var boxRight = centerX + (LastDetectionBox.Width / 2.0);

                    switch (tracerPosition)
                    {
                        case "Top":
                            DetectedPlayerOverlay.DetectedTracers.X2 = boxHorizontalCenter;
                            DetectedPlayerOverlay.DetectedTracers.Y2 = boxTop;
                            break;

                        case "Bottom":
                            DetectedPlayerOverlay.DetectedTracers.X2 = boxHorizontalCenter;
                            DetectedPlayerOverlay.DetectedTracers.Y2 = boxBottom;
                            break;

                        case "Middle":
                            var screenHorizontalCenter = DisplayManager.ScreenWidth / (2.0 * WinAPICaller.scalingFactorX);
                            if (boxHorizontalCenter < screenHorizontalCenter)
                            {
                                // if the box is on the left half of the screen, aim for the right-middle of the box
                                DetectedPlayerOverlay.DetectedTracers.X2 = boxRight;
                                DetectedPlayerOverlay.DetectedTracers.Y2 = boxVerticalCenter;
                            }
                            else
                            {
                                // if the box is on the right half, aim for the left-middle
                                DetectedPlayerOverlay.DetectedTracers.X2 = boxLeft;
                                DetectedPlayerOverlay.DetectedTracers.Y2 = boxVerticalCenter;
                            }
                            break;

                        default:
                            // default to the bottom-center if the setting is unrecognized
                            DetectedPlayerOverlay.DetectedTracers.X2 = boxHorizontalCenter;
                            DetectedPlayerOverlay.DetectedTracers.Y2 = boxBottom;
                            break;
                    }
                }

                DetectedPlayerOverlay.Opacity = Dictionary.sliderSettings["Opacity"];

                DetectedPlayerOverlay.DetectedPlayerFocus.Opacity = 1;
                DetectedPlayerOverlay.DetectedPlayerFocus.Margin = new Thickness(
                    centerX - (LastDetectionBox.Width / 2.0), centerY, 0, 0);
                DetectedPlayerOverlay.DetectedPlayerFocus.Width = LastDetectionBox.Width;
                DetectedPlayerOverlay.DetectedPlayerFocus.Height = LastDetectionBox.Height;
            });
        }

        private void CalculateCoordinates(DetectedPlayerWindow DetectedPlayerOverlay, Prediction closestPrediction, float scaleX, float scaleY)
        {
            AIConf = closestPrediction.Confidence;

            if (Dictionary.toggleState["Show Detected Player"] && Dictionary.DetectedPlayerOverlay != null)
            {
                using (Benchmark("UpdateOverlay"))
                {
                    UpdateOverlay(DetectedPlayerOverlay!, closestPrediction);
                }
                if (!Dictionary.toggleState["Aim Assist"]) return;
            }

            double YOffset = Dictionary.sliderSettings["Y Offset (Up/Down)"];
            double XOffset = Dictionary.sliderSettings["X Offset (Left/Right)"];

            double YOffsetPercentage = Dictionary.sliderSettings["Y Offset (%)"];
            double XOffsetPercentage = Dictionary.sliderSettings["X Offset (%)"];

            var rect = closestPrediction.Rectangle;

            if (Dictionary.toggleState["X Axis Percentage Adjustment"])
            {
                detectedX = (int)((rect.X + (rect.Width * (XOffsetPercentage / 100))) * scaleX);
            }
            else
            {
                detectedX = (int)((rect.X + rect.Width / 2) * scaleX + XOffset);
            }

            if (Dictionary.toggleState["Y Axis Percentage Adjustment"])
            {
                detectedY = (int)((rect.Y + rect.Height - (rect.Height * (YOffsetPercentage / 100))) * scaleY + YOffset);
            }
            else
            {
                detectedY = CalculateDetectedY(scaleY, YOffset, closestPrediction);
            }

            if (Dictionary.toggleState["Prediction Pixel Offset Enabled"])
            {
                double maxOffset = Convert.ToDouble(Dictionary.sliderSettings["Prediction Pixel Offset"]);
                int screenCenterX = ScreenWidth / 2;
                int screenCenterY = ScreenHeight / 2;

                // Box boundaries in screen coordinates
                float boxLeft = rect.X * scaleX;
                float boxRight = (rect.X + rect.Width) * scaleX;
                float boxTop = rect.Y * scaleY;
                float boxBottom = (rect.Y + rect.Height) * scaleY;

                // Find the pixel point inside the target box that is closest to screen center (crosshair)
                double closestPixelX = Math.Clamp(screenCenterX, boxLeft, boxRight);
                double closestPixelY = Math.Clamp(screenCenterY, boxTop, boxBottom);

                // Offset from base detected point towards the closest pixel on target, bounded by maxOffset
                double dirX = closestPixelX - detectedX;
                double dirY = closestPixelY - detectedY;
                double dist = Math.Sqrt(dirX * dirX + dirY * dirY);

                if (dist > 0)
                {
                    double appliedOffset = Math.Min(dist, maxOffset);
                    detectedX = (int)(detectedX + (dirX / dist) * appliedOffset);
                    detectedY = (int)(detectedY + (dirY / dist) * appliedOffset);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CalculateDetectedY(float scaleY, double YOffset, Prediction closestPrediction)
        {
            var rect = closestPrediction.Rectangle;
            float yBase = rect.Y;
            float yAdjustment = 0;

            switch (Dictionary.dropdownState["Aiming Boundaries Alignment"])
            {
                case "Center":
                    yAdjustment = rect.Height / 2;
                    break;

                case "Top":
                    // yBase is already at the top
                    break;

                case "Bottom":
                    yAdjustment = rect.Height;
                    break;
            }

            return (int)((yBase + yAdjustment) * scaleY + YOffset);
        }

        private static bool IsInsideFovAction(Prediction prediction, float center, float actionRadius, string fovStyle)
        {
            var rect = prediction.Rectangle;
            // Closest point on the target's bounding box to the center of the FOV (image center)
            float closestX = Math.Clamp(center, rect.X, rect.X + rect.Width);
            float closestY = Math.Clamp(center, rect.Y, rect.Y + rect.Height);

            if (fovStyle == "Rectangle")
            {
                return Math.Abs(closestX - center) <= actionRadius && Math.Abs(closestY - center) <= actionRadius;
            }
            else // Circle
            {
                float dx = closestX - center;
                float dy = closestY - center;
                return (dx * dx + dy * dy) <= (actionRadius * actionRadius);
            }
        }

        private void HandleAim(Prediction closestPrediction)
        {
            // Rate-limit aim commands to prevent the AI loop from outrunning the display.
            // DirectX loop can run at 200+ FPS; sending mouse commands that fast causes
            // overshoot-undershoot oscillation because the game hasn't rendered the result
            // of the previous command yet.
            long elapsed = _aimRateTimer.ElapsedMilliseconds;
            if (elapsed < AIM_MIN_INTERVAL_MS)
            {
                return;
            }
            _aimRateTimer.Restart();

            if (Dictionary.toggleState["FOV Action"])
            {
                float actionSize = (float)Dictionary.sliderSettings["FOV Action Size"];
                float center = IMAGE_SIZE / 2.0f;
                string fovStyle = Dictionary.dropdownState.TryGetValue("FOV Style", out var s) ? s.ToString() : "Circle";
                if (!IsInsideFovAction(closestPrediction, center, actionSize / 2.0f, fovStyle))
                {
                    MouseManager.ResetMovement();
                    return;
                }
            }

            if (Dictionary.toggleState["Insta-Flick"] && (InputBindingManager.IsHoldingBinding("Insta-Flick Keybind") || InputBindingManager.IsHoldingBinding("Second Insta-Flick Keybind")))
            {
                MouseManager.InstaFlick(detectedX, detectedY);
                return;
            }

            if (Dictionary.toggleState["Aim Assist"] &&
                (Dictionary.toggleState["Constant AI Tracking"] ||
                 Dictionary.toggleState["Aim Assist"] && InputBindingManager.IsHoldingBinding("Aim Keybind") ||
                 Dictionary.toggleState["Aim Assist"] && InputBindingManager.IsHoldingBinding("Second Aim Keybind")))
            {
                if (Dictionary.toggleState["Predictions"])
                {
                    HandlePredictions(kalmanPrediction, closestPrediction, detectedX, detectedY);
                }
                else
                {
                    MouseManager.MoveCrosshair(detectedX, detectedY);
                }
            }
            else
            {
                MouseManager.ResetMovement();
            }
        }

        private void HandlePredictions(KalmanPrediction kalmanPrediction, Prediction closestPrediction, int detectedX, int detectedY)
        {
            int predX = detectedX;
            int predY = detectedY;

            var predictionMethod = Dictionary.dropdownState["Prediction Method"];
            switch (predictionMethod)
            {
                case "Kalman Filter":
                    KalmanPrediction.Detection detection = new()
                    {
                        X = detectedX,
                        Y = detectedY,
                        Timestamp = DateTime.UtcNow
                    };

                    kalmanPrediction.UpdateKalmanFilter(detection);
                    var predictedPosition = kalmanPrediction.GetKalmanPosition();
                    predX = predictedPosition.X;
                    predY = predictedPosition.Y;
                    break;

                case "Shall0e's Prediction":
                    // Update position (calculates velocity internally)
                    ShalloePredictionV2.UpdatePosition(detectedX, detectedY);

                    // Get predicted position
                    predX = ShalloePredictionV2.GetSPX();
                    predY = ShalloePredictionV2.GetSPY();
                    break;

                case "wisethef0x's EMA Prediction":
                    WiseTheFoxPrediction.WTFDetection wtfdetection = new()
                    {
                        X = detectedX,
                        Y = detectedY,
                        Timestamp = DateTime.UtcNow
                    };

                    wtfpredictionManager.UpdateDetection(wtfdetection);
                    var wtfpredictedPosition = wtfpredictionManager.GetEstimatedPosition();

                    // Use both predicted X and Y
                    predX = wtfpredictedPosition.X;
                    predY = wtfpredictedPosition.Y;
                    break;
            }

            if (Dictionary.toggleState["Flick Aim"])
            {
                int deltaX = predX - detectedX;
                int deltaY = predY - detectedY;
                double flickStrength = Convert.ToDouble(Dictionary.sliderSettings["Flick Aim Strength"]);
                predX = (int)(detectedX + deltaX * (1.0 + flickStrength * 2.0));
                predY = (int)(detectedY + deltaY * (1.0 + flickStrength * 2.0));
            }

            if (Dictionary.toggleState["Prediction Pixel Offset Enabled"])
            {
                double maxOffset = Convert.ToDouble(Dictionary.sliderSettings["Prediction Pixel Offset"]);
                int pDeltaX = predX - detectedX;
                int pDeltaY = predY - detectedY;
                double dist = Math.Sqrt(pDeltaX * pDeltaX + pDeltaY * pDeltaY);
                if (dist > maxOffset && dist > 0)
                {
                    predX = detectedX + (int)(pDeltaX / dist * maxOffset);
                    predY = detectedY + (int)(pDeltaY / dist * maxOffset);
                }
            }

            MouseManager.MoveCrosshair(predX, predY);
        }

        private async Task<(Prediction? Prediction, bool FrameAcquired)> GetClosestPrediction(bool useMousePosition = true)
        {
            //whats these variables for? - taylor 
            //int adjustedTargetX, adjustedTargetY;

            if (Dictionary.dropdownState["Detection Area Type"] == "Closest to Mouse")
            {
                var mousePos = WinAPICaller.GetCursorPosition();

                // Check if mouse is on the current display
                if (DisplayManager.IsPointInCurrentDisplay(new System.Windows.Point(mousePos.X, mousePos.Y)))
                {
                    // Mouse is on current display, use its position
                    targetX = mousePos.X;
                    targetY = mousePos.Y;
                }
                else
                {
                    // Mouse is on different display, use center of current display
                    targetX = DisplayManager.ScreenLeft + (DisplayManager.ScreenWidth / 2);
                    targetY = DisplayManager.ScreenTop + (DisplayManager.ScreenHeight / 2);
                }
            }
            else
            {
                // Center of current display
                targetX = DisplayManager.ScreenLeft + (DisplayManager.ScreenWidth / 2);
                targetY = DisplayManager.ScreenTop + (DisplayManager.ScreenHeight / 2);
            }

            Rectangle detectionBox = new(targetX - IMAGE_SIZE / 2, targetY - IMAGE_SIZE / 2, IMAGE_SIZE, IMAGE_SIZE); // Detection box dynamic size

            Bitmap? frame = null;

            using (Benchmark("ScreenGrab"))
            {
                frame = _captureManager.ScreenGrab(detectionBox);
            }

            if (frame == null) return (null, false);

            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
            Tensor<float> outputTensor;

            float[] inputArray;
            using (Benchmark("BitmapToFloatArray"))
            {
                if (_reusableInputArray == null || _reusableInputArray.Length != 3 * IMAGE_SIZE * IMAGE_SIZE)
                {
                    _reusableInputArray = new float[3 * IMAGE_SIZE * IMAGE_SIZE];
                }
                inputArray = _reusableInputArray;

                // Fill the reusable array
                BitmapToFloatArrayInPlace(frame, inputArray, IMAGE_SIZE);
            }

            // Reuse tensor and inputs - recreate if size changed
            /// this needs to be revised !!!!! - taylor
            if (_reusableTensor == null || _reusableTensor.Dimensions[2] != IMAGE_SIZE)
            {
                _reusableTensor = new DenseTensor<float>(inputArray, new int[] { 1, 3, IMAGE_SIZE, IMAGE_SIZE });
                _reusableInputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, _reusableTensor) };
            }
            else
            {
                // Directly copy into existing DenseTensor buffer
                inputArray.AsSpan().CopyTo(_reusableTensor.Buffer.Span);
            }

            using (Benchmark("ModelInference"))
            {
                results = _onnxModel.Run(_reusableInputs, _outputNames, _modeloptions);
                outputTensor = results[0].AsTensor<float>();
            }

            if (outputTensor == null)
            {
                Log(LogLevel.Error, "Model inference returned null output tensor.", true, 2000);
                SaveFrame(frame);
                frame.Dispose();
                results.Dispose();
                return (null, true);
            }

            // Calculate the FOV boundaries (Detection uses the full FOV Size so enemies in FOV are detected and displayed)
            float FovSize = (float)Dictionary.sliderSettings["FOV Size"];
            float fovMinX = (IMAGE_SIZE - FovSize) / 2.0f;
            float fovMaxX = (IMAGE_SIZE + FovSize) / 2.0f;
            float fovMinY = (IMAGE_SIZE - FovSize) / 2.0f;
            float fovMaxY = (IMAGE_SIZE + FovSize) / 2.0f;

            //List<double[]> KDpoints;
            List<Prediction> KDPredictions;
            using (Benchmark("PrepareKDTreeData"))
            {
                KDPredictions = PrepareKDTreeData(outputTensor, detectionBox, fovMinX, fovMaxX, fovMinY, fovMaxY);
            }

            results.Dispose();

            if (KDPredictions.Count == 0)
            {
                SaveFrame(frame);
                frame.Dispose();
                return (null, true);
            }

            //kdtree was replaced with linear search
            Prediction? bestCandidate = null;
            double bestDistSq = double.MaxValue;
            double center = IMAGE_SIZE / 2.0;

            bool fovActionEnabled = Dictionary.toggleState["FOV Action"];
            float actionSize = (float)Dictionary.sliderSettings["FOV Action Size"];
            float actionRadius = actionSize / 2.0f;
            string fovStyle = Dictionary.dropdownState.TryGetValue("FOV Style", out var fs) ? fs.ToString() : "Circle";

            Prediction? bestActionCandidate = null;
            double bestActionDistSq = double.MaxValue;

            // TODO: Optimize this linear search further if needed
            // TODO: Consider updating KD-Tree and adding options to switch from linear to kd.
            // we can honestly replacing linear search by letting sticky aim handle the search
            using (Benchmark("LinearSearch"))
            {
                foreach (var p in KDPredictions)
                {
                    var dx = p.CenterXTranslated * IMAGE_SIZE - center;
                    var dy = p.CenterYTranslated * IMAGE_SIZE - center;
                    double d2 = dx * dx + dy * dy; // dx^2 + dy^2

                    if (d2 < bestDistSq) { bestDistSq = d2; bestCandidate = p; }

                    if (fovActionEnabled && IsInsideFovAction(p, (float)center, actionRadius, fovStyle))
                    {
                        if (d2 < bestActionDistSq) { bestActionDistSq = d2; bestActionCandidate = p; }
                    }
                }
            }

            // If FOV Action is active and we have a target inside the action area, prioritize it!
            if (fovActionEnabled && bestActionCandidate != null)
            {
                bestCandidate = bestActionCandidate;
            }

            Prediction? finalTarget = HandleStickyAim(bestCandidate, KDPredictions);
            if (finalTarget != null)
            {
                UpdateDetectionBox(finalTarget, detectionBox);
                SaveFrame(frame, finalTarget);
                frame.Dispose();
                return (finalTarget, true);
            }

            frame.Dispose();
            return (null, true);
        }

        private Prediction? HandleStickyAim(Prediction? bestCandidate, List<Prediction> KDPredictions)
        {
            if (!Dictionary.toggleState["Sticky Aim"])
            {
                _currentTarget = bestCandidate;
                ResetStickyAimState();
                return bestCandidate;
            }

            // No detections available
            if (bestCandidate == null || KDPredictions == null || KDPredictions.Count == 0)
            {
                return HandleNoDetections();
            }

            // Center of the frame (where the crosshair is located in IMAGE_SIZE space)
            float centerX = IMAGE_SIZE / 2f;
            float centerY = IMAGE_SIZE / 2f;

            // If we don't have an active target locked, acquire bestCandidate immediately
            if (_currentTarget == null)
            {
                return AcquireNewTarget(bestCandidate);
            }

            // STEP 1: Find if the currently locked target still exists in the current detections
            float lastX = _currentTarget.CenterXTranslated * IMAGE_SIZE;
            float lastY = _currentTarget.CenterYTranslated * IMAGE_SIZE;
            float expectedX = lastX + _lastTargetVelocityX;
            float expectedY = lastY + _lastTargetVelocityY;

            float targetSize = MathF.Max(_currentTarget.Rectangle.Width, _currentTarget.Rectangle.Height);
            float maxMatchDist = Math.Clamp(targetSize * 3.0f, 80f, 250f);
            float maxMatchDistSq = maxMatchDist * maxMatchDist;

            Prediction? matchedTarget = null;
            float bestMatchDistSq = float.MaxValue;

            foreach (var candidate in KDPredictions)
            {
                float candX = candidate.CenterXTranslated * IMAGE_SIZE;
                float candY = candidate.CenterYTranslated * IMAGE_SIZE;
                float dSq = GetDistanceSq(candX, candY, expectedX, expectedY);
                if (dSq < bestMatchDistSq && dSq < maxMatchDistSq)
                {
                    bestMatchDistSq = dSq;
                    matchedTarget = candidate;
                }
            }

            // STEP 2: Handle if current target is not detected in this frame
            if (matchedTarget == null)
            {
                // Target lost - switch smoothly to best candidate
                return AcquireNewTarget(bestCandidate);
            }

            // Current target matched! Reset consecutive frames counter and update velocity
            _consecutiveFramesWithoutTarget = 0;
            float currX = matchedTarget.CenterXTranslated * IMAGE_SIZE;
            float currY = matchedTarget.CenterYTranslated * IMAGE_SIZE;
            _lastTargetVelocityX = _lastTargetVelocityX * 0.6f + (currX - lastX) * 0.4f;
            _lastTargetVelocityY = _lastTargetVelocityY * 0.6f + (currY - lastY) * 0.4f;

            // If the matched target is already the best candidate, continue tracking it
            if (ReferenceEquals(matchedTarget, bestCandidate))
            {
                _currentTarget = matchedTarget;
                return matchedTarget;
            }

            // STEP 3: A different candidate is closer to the crosshair.
            // Check FOV Action first: if current target left the FOV Action zone while bestCandidate is inside, switch immediately.
            bool fovActionEnabled = Dictionary.toggleState["FOV Action"];
            if (fovActionEnabled)
            {
                float actionSize = (float)Dictionary.sliderSettings["FOV Action Size"];
                float actionRadius = actionSize / 2.0f;
                string fovStyle = Dictionary.dropdownState.TryGetValue("FOV Style", out var fs) ? fs.ToString() : "Circle";

                bool currentInAction = IsInsideFovAction(matchedTarget, centerX, actionRadius, fovStyle);
                bool bestInAction = IsInsideFovAction(bestCandidate, centerX, actionRadius, fovStyle);

                if (!currentInAction && bestInAction)
                {
                    return AcquireNewTarget(bestCandidate);
                }
            }

            // STEP 4: Sticky Aim Detection threshold logic
            // "Low threshold: smaller detection pixel area, meaning if a new detection appears it will move to next closest target.
            //  High threshold: full pixel detection -> aim will stay sticky on target."
            float threshold = (float)Dictionary.sliderSettings["Sticky Aim Threshold"];

            // High threshold (near 100): Full pixel detection -> stays sticky on current target across the full detection area
            if (threshold >= 98f)
            {
                _currentTarget = matchedTarget;
                return matchedTarget;
            }

            // Low threshold (near 0): Stickiness disabled/minimal -> immediately move to next closest target
            if (threshold <= 2f)
            {
                return AcquireNewTarget(bestCandidate);
            }

            // Intermediate threshold: Define the sticky detection pixel area around the crosshair
            // Detection area scale from 0 to full detection radius (up to IMAGE_SIZE / 2)
            float maxDetectionRadius = Math.Max(IMAGE_SIZE / 2f, (float)Dictionary.sliderSettings["FOV Size"] / 2f);
            float stickyDetectionRadius = (threshold / 100f) * (maxDetectionRadius * 1.4f);

            float currentDist = MathF.Sqrt(GetDistanceSq(currX, currY, centerX, centerY));
            float newDist = MathF.Sqrt(GetDistanceSq(bestCandidate.CenterXTranslated * IMAGE_SIZE, bestCandidate.CenterYTranslated * IMAGE_SIZE, centerX, centerY));

            // If current target has moved outside the sticky detection pixel area, switch to the new closest target
            if (currentDist > stickyDetectionRadius)
            {
                return AcquireNewTarget(bestCandidate);
            }

            // Current target is inside the sticky detection area.
            // Require the new target to be closer by a margin proportional to the threshold
            float stickyMargin = stickyDetectionRadius * 0.6f;
            if (newDist + stickyMargin < currentDist)
            {
                return AcquireNewTarget(bestCandidate);
            }

            // Otherwise, stay sticky on current target
            _currentTarget = matchedTarget;
            return matchedTarget;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float GetDistanceSq(float x1, float y1, float x2, float y2)
        {
            float dx = x1 - x2;
            float dy = y1 - y2;
            return dx * dx + dy * dy;
        }

        private Prediction? HandleNoDetections()
        {
            if (_currentTarget != null && ++_consecutiveFramesWithoutTarget <= MAX_FRAMES_WITHOUT_TARGET)
            {
                // Decay velocity during grace period
                _lastTargetVelocityX *= 0.8f;
                _lastTargetVelocityY *= 0.8f;

                // Return predicted position instead of stale position
                var predicted = new Prediction
                {
                    ScreenCenterX = _currentTarget.ScreenCenterX + _lastTargetVelocityX,
                    ScreenCenterY = _currentTarget.ScreenCenterY + _lastTargetVelocityY,
                    Rectangle = _currentTarget.Rectangle,
                    Confidence = _currentTarget.Confidence * (1f - _consecutiveFramesWithoutTarget * 0.2f),
                    ClassId = _currentTarget.ClassId,
                    ClassName = _currentTarget.ClassName,
                    CenterXTranslated = _currentTarget.CenterXTranslated + (_lastTargetVelocityX / IMAGE_SIZE),
                    CenterYTranslated = _currentTarget.CenterYTranslated + (_lastTargetVelocityY / IMAGE_SIZE)
                };
                return predicted;
            }

            ResetStickyAimState();
            return null;
        }

        private Prediction AcquireNewTarget(Prediction target)
        {
            _lastTargetVelocityX = 0f;
            _lastTargetVelocityY = 0f;
            _consecutiveFramesWithoutTarget = 0;
            _currentTarget = target;
            return target;
        }

        private void ResetStickyAimState()
        {
            _currentTarget = null;
            _consecutiveFramesWithoutTarget = 0;
            _lastTargetVelocityX = 0f;
            _lastTargetVelocityY = 0f;
        }

        private void UpdateDetectionBox(Prediction target, Rectangle detectionBox)
        {
            float translatedXMin = target.Rectangle.X + detectionBox.Left;
            float translatedYMin = target.Rectangle.Y + detectionBox.Top;
            LastDetectionBox = new(translatedXMin, translatedYMin,
                target.Rectangle.Width, target.Rectangle.Height);

            CenterXTranslated = target.CenterXTranslated;
            CenterYTranslated = target.CenterYTranslated;
        }
        // is it really kdtreedata though....
        private List<Prediction> PrepareKDTreeData(
            Tensor<float> outputTensor,
            Rectangle detectionBox,
            float fovMinX, float fovMaxX, float fovMinY, float fovMaxY)
        {
            float minConfidence = (float)Dictionary.sliderSettings["AI Minimum Confidence"] / 100.0f;
            string selectedClass = Dictionary.dropdownState["Target Class"];
            int selectedClassId = selectedClass == "Best Confidence" ? -1 : _modelClasses.FirstOrDefault(c => c.Value == selectedClass).Key;

            var KDpredictions = new List<Prediction>(NUM_DETECTIONS);

            if (outputTensor.Dimensions.Length != 3)
            {
                return KDpredictions;
            }

            int dim1 = outputTensor.Dimensions[1];
            int dim2 = outputTensor.Dimensions[2];

            // Format 1: YOLOv26 / YOLOv10 End-to-End: [1, N, 6] (x1, y1, x2, y2, confidence, class_id)
            if (_outputFormat == ModelOutputFormat.YOLO_EndToEnd_BoxesFirst || dim2 == 6)
            {
                for (int i = 0; i < dim1; i++)
                {
                    float x1 = outputTensor[0, i, 0];
                    float y1 = outputTensor[0, i, 1];
                    float x2 = outputTensor[0, i, 2];
                    float y2 = outputTensor[0, i, 3];
                    float confidence = outputTensor[0, i, 4];
                    int classId = (int)outputTensor[0, i, 5];

                    if (confidence < minConfidence) continue;
                    if (selectedClassId != -1 && classId != selectedClassId) continue;

                    // If coordinates are normalized [0..1], rescale to IMAGE_SIZE
                    if (x2 <= 1.0f && y2 <= 1.0f && (x2 > 0 || y2 > 0))
                    {
                        x1 *= IMAGE_SIZE;
                        y1 *= IMAGE_SIZE;
                        x2 *= IMAGE_SIZE;
                        y2 *= IMAGE_SIZE;
                    }

                    float width = x2 - x1;
                    float height = y2 - y1;
                    if (width <= 0 || height <= 0) continue;

                    float x_center = x1 + width / 2.0f;
                    float y_center = y1 + height / 2.0f;

                    if (x1 < fovMinX || x2 > fovMaxX || y1 < fovMinY || y2 > fovMaxY) continue;

                    RectangleF rect = new(x1, y1, width, height);
                    Prediction prediction = new()
                    {
                        Rectangle = rect,
                        Confidence = confidence,
                        ClassId = classId,
                        ClassName = _modelClasses.GetValueOrDefault(classId, $"Class_{classId}"),
                        CenterXTranslated = x_center / IMAGE_SIZE,
                        CenterYTranslated = y_center / IMAGE_SIZE,
                        ScreenCenterX = detectionBox.Left + x_center,
                        ScreenCenterY = detectionBox.Top + y_center
                    };

                    KDpredictions.Add(prediction);
                }
            }
            // Format 2: Transposed End-to-End: [1, 6, N]
            else if (_outputFormat == ModelOutputFormat.YOLO_EndToEnd_ChannelsFirst || dim1 == 6)
            {
                for (int i = 0; i < dim2; i++)
                {
                    float x1 = outputTensor[0, 0, i];
                    float y1 = outputTensor[0, 1, i];
                    float x2 = outputTensor[0, 2, i];
                    float y2 = outputTensor[0, 3, i];
                    float confidence = outputTensor[0, 4, i];
                    int classId = (int)outputTensor[0, 5, i];

                    if (confidence < minConfidence) continue;
                    if (selectedClassId != -1 && classId != selectedClassId) continue;

                    if (x2 <= 1.0f && y2 <= 1.0f && (x2 > 0 || y2 > 0))
                    {
                        x1 *= IMAGE_SIZE;
                        y1 *= IMAGE_SIZE;
                        x2 *= IMAGE_SIZE;
                        y2 *= IMAGE_SIZE;
                    }

                    float width = x2 - x1;
                    float height = y2 - y1;
                    if (width <= 0 || height <= 0) continue;

                    float x_center = x1 + width / 2.0f;
                    float y_center = y1 + height / 2.0f;

                    if (x1 < fovMinX || x2 > fovMaxX || y1 < fovMinY || y2 > fovMaxY) continue;

                    RectangleF rect = new(x1, y1, width, height);
                    Prediction prediction = new()
                    {
                        Rectangle = rect,
                        Confidence = confidence,
                        ClassId = classId,
                        ClassName = _modelClasses.GetValueOrDefault(classId, $"Class_{classId}"),
                        CenterXTranslated = x_center / IMAGE_SIZE,
                        CenterYTranslated = y_center / IMAGE_SIZE,
                        ScreenCenterX = detectionBox.Left + x_center,
                        ScreenCenterY = detectionBox.Top + y_center
                    };

                    KDpredictions.Add(prediction);
                }
            }
            // Format 3: Transposed raw YOLO: [1, N, 4 + NUM_CLASSES]
            else if (_outputFormat == ModelOutputFormat.YOLO_Raw_ChannelsLast || (dim1 > dim2 && dim2 != 6))
            {
                int numClasses = dim2 - 4;
                for (int i = 0; i < dim1; i++)
                {
                    float x_center = outputTensor[0, i, 0];
                    float y_center = outputTensor[0, i, 1];
                    float width = outputTensor[0, i, 2];
                    float height = outputTensor[0, i, 3];

                    int bestClassId = 0;
                    float bestConfidence = 0f;

                    if (numClasses <= 1)
                    {
                        bestConfidence = outputTensor[0, i, 4];
                    }
                    else
                    {
                        if (selectedClassId == -1)
                        {
                            for (int classId = 0; classId < numClasses; classId++)
                            {
                                float classConfidence = outputTensor[0, i, 4 + classId];
                                if (classConfidence > bestConfidence)
                                {
                                    bestConfidence = classConfidence;
                                    bestClassId = classId;
                                }
                            }
                        }
                        else
                        {
                            bestConfidence = outputTensor[0, i, 4 + selectedClassId];
                            bestClassId = selectedClassId;
                        }
                    }

                    if (bestConfidence < minConfidence) continue;

                    float x_min = x_center - width / 2;
                    float y_min = y_center - height / 2;
                    float x_max = x_center + width / 2;
                    float y_max = y_center + height / 2;

                    if (x_min < fovMinX || x_max > fovMaxX || y_min < fovMinY || y_max > fovMaxY) continue;

                    RectangleF rect = new(x_min, y_min, width, height);
                    Prediction prediction = new()
                    {
                        Rectangle = rect,
                        Confidence = bestConfidence,
                        ClassId = bestClassId,
                        ClassName = _modelClasses.GetValueOrDefault(bestClassId, $"Class_{bestClassId}"),
                        CenterXTranslated = x_center / IMAGE_SIZE,
                        CenterYTranslated = y_center / IMAGE_SIZE,
                        ScreenCenterX = detectionBox.Left + x_center,
                        ScreenCenterY = detectionBox.Top + y_center
                    };

                    KDpredictions.Add(prediction);
                }
            }
            // Format 4: Default YOLOv8/v9/v11/v26 raw: [1, 4 + NUM_CLASSES, N]
            else
            {
                int numDetections = dim2;
                for (int i = 0; i < numDetections; i++)
                {
                    float x_center = outputTensor[0, 0, i];
                    float y_center = outputTensor[0, 1, i];
                    float width = outputTensor[0, 2, i];
                    float height = outputTensor[0, 3, i];

                    int bestClassId = 0;
                    float bestConfidence = 0f;

                    if (NUM_CLASSES == 1)
                    {
                        bestConfidence = outputTensor[0, 4, i];
                    }
                    else
                    {
                        if (selectedClassId == -1)
                        {
                            for (int classId = 0; classId < NUM_CLASSES; classId++)
                            {
                                float classConfidence = outputTensor[0, 4 + classId, i];
                                if (classConfidence > bestConfidence)
                                {
                                    bestConfidence = classConfidence;
                                    bestClassId = classId;
                                }
                            }
                        }
                        else
                        {
                            bestConfidence = outputTensor[0, 4 + selectedClassId, i];
                            bestClassId = selectedClassId;
                        }
                    }

                    if (bestConfidence < minConfidence) continue;

                    float x_min = x_center - width / 2;
                    float y_min = y_center - height / 2;
                    float x_max = x_center + width / 2;
                    float y_max = y_center + height / 2;

                    if (x_min < fovMinX || x_max > fovMaxX || y_min < fovMinY || y_max > fovMaxY) continue;

                    RectangleF rect = new(x_min, y_min, width, height);
                    Prediction prediction = new()
                    {
                        Rectangle = rect,
                        Confidence = bestConfidence,
                        ClassId = bestClassId,
                        ClassName = _modelClasses.GetValueOrDefault(bestClassId, $"Class_{bestClassId}"),
                        CenterXTranslated = x_center / IMAGE_SIZE,
                        CenterYTranslated = y_center / IMAGE_SIZE,
                        ScreenCenterX = detectionBox.Left + x_center,
                        ScreenCenterY = detectionBox.Top + y_center
                    };

                    KDpredictions.Add(prediction);
                }
            }

            return KDpredictions;
        }

        #endregion AI Loop Functions

        #endregion AI

        #region Screen Capture

        private void SaveFrame(Bitmap frame, Prediction? DoLabel = null)
        {
            // Only save frames if "Collect Data While Playing" is enabled
            if (!Dictionary.toggleState["Collect Data While Playing"]) return;

            // Skip if we're in constant tracking mode (unless auto-labeling is enabled)
            if (Dictionary.toggleState["Constant AI Tracking"] && !Dictionary.toggleState["Auto Label Data"]) return;

            // Cooldown check
            if ((DateTime.Now - lastSavedTime).TotalMilliseconds < SAVE_FRAME_COOLDOWN_MS) return;

            try
            {
                // Validate bitmap is still usable
                if (frame == null) return;

                // Accessing Width/Height will throw if bitmap is disposed
                int width = frame.Width;
                int height = frame.Height;
                if (width <= 0 || height <= 0) return;

                lastSavedTime = DateTime.Now;
                string uuid = Guid.NewGuid().ToString();
                string imagePath = Path.Combine("bin", "images", $"{uuid}.jpg");

                // Save synchronously to avoid "Object is currently in use elsewhere" error
                frame.Save(imagePath, ImageFormat.Jpeg);

                if (Dictionary.toggleState["Auto Label Data"] && DoLabel != null)
                {
                    var labelPath = Path.Combine("bin", "labels", $"{uuid}.txt");

                    float x = (DoLabel!.Rectangle.X + DoLabel.Rectangle.Width / 2) / width;
                    float y = (DoLabel!.Rectangle.Y + DoLabel.Rectangle.Height / 2) / height;
                    float labelWidth = DoLabel.Rectangle.Width / width;
                    float labelHeight = DoLabel.Rectangle.Height / height;

                    File.WriteAllText(labelPath, $"{DoLabel.ClassId} {x} {y} {labelWidth} {labelHeight}");
                }
            }
            catch (ArgumentException)
            {
                // Bitmap was disposed or invalid - silently ignore
            }
            catch (Exception ex)
            {
                Log(LogLevel.Error, $"SaveFrame failed: {ex.Message}");
            }
        }



        #endregion Screen Capture

        public void Dispose()
        {
            // Signal that we're shutting down
            lock (_sizeLock)
            {
                _sizeChangePending = true;
            }

            // Stop the loop
            _isAiLoopRunning = false;
            MouseManager.ResetMovement();

            if (_aiLoopThread != null && _aiLoopThread.IsAlive)
            {
                _aiLoopThread.Join();
            }

            // Print final benchmarks
            PrintBenchmarks();

            // Dispose DXGI objects
            _captureManager.Dispose();

            // Clean up other resources
            _reusableInputArray = null;
            _reusableInputs = null;
            _onnxModel?.Dispose();
            _modeloptions?.Dispose();
            _onnxModel = null;
            _bitmapBuffer = null;
        }
    }
    public class Prediction
    {
        public RectangleF Rectangle { get; set; }
        public float Confidence { get; set; }
        public int ClassId { get; set; } = 0;
        public string ClassName { get; set; } = "Enemy";
        public float CenterXTranslated { get; set; }
        public float CenterYTranslated { get; set; }
        public float ScreenCenterX { get; set; }  // Absolute screen position
        public float ScreenCenterY { get; set; }
    }

    public class DataStreamTransformer
    {
        private int _seed;
        private int[] _internalBuffer;
        private string _stateHash;

        public DataStreamTransformer()
        {
            _seed = 1024;
            _internalBuffer = new int[12];
            _stateHash = "init_segment";
            InitializeBuffer();
        }

        private void InitializeBuffer()
        {
            for (int i = 0; i < _internalBuffer.Length; i++)
            {
                _internalBuffer[i] = (i * 17) ^ (_seed + i);
            }
        }

        public int RunTransformation(int factor, int rounds)
        {
            int localAccumulator = _seed;
            int dynamicOffset = (factor & 0x0F) + 1;

            for (int i = 0; i < rounds; i++)
            {
                if (i % 2 == 0)
                {
                    localAccumulator += (_internalBuffer[i % _internalBuffer.Length] ^ dynamicOffset);
                    _stateHash += (i * factor).ToString();
                }
                else
                {
                    localAccumulator -= (dynamicOffset * 3) ^ (i + 7);
                    dynamicOffset = (dynamicOffset << 1) ^ (localAccumulator & 0x03);
                }

                switch (i % 4)
                {
                    case 0:
                        localAccumulator = (localAccumulator * 31) + 11;
                        break;
                    case 1:
                        localAccumulator = (localAccumulator >> 1) ^ 0x5A5A;
                        break;
                    case 2:
                        localAccumulator ^= (dynamicOffset + i);
                        break;
                    default:
                        localAccumulator += (_seed % (i + 1));
                        break;
                }

                for (int j = 0; j < _internalBuffer.Length; j++)
                {
                    _internalBuffer[j] = (_internalBuffer[j] + localAccumulator) ^ (dynamicOffset * 2);
                }
            }

            // Opaque predicate: n * (n + 1) is always even
            int verification = localAccumulator * (localAccumulator + 1);
            if (verification % 2 != 0)
            {
                return FallbackCalculation(localAccumulator);
            }

            return localAccumulator ^ _internalBuffer[0];
        }

        private int FallbackCalculation(int fallbackValue)
        {
            int surrogate = fallbackValue;
            for (int k = 0; k < _internalBuffer.Length; k++)
            {
                surrogate += _internalBuffer[k] * 2;
            }
            return surrogate;
        }
    }
}