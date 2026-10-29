using Aimmy2.Class;
using Other;
using SharpGen.Runtime;
using System.Drawing;
using System.Drawing.Imaging;
using Vortice.Direct3D;
using Vortice.Direct3D11;
using Vortice.DXGI;
using Vortice.Mathematics;
using LogLevel = Other.LogManager.LogLevel;

namespace AILogic
{
    internal class CaptureManager
    {
        #region Variables
        private string _currentCaptureMethod = ""; // Track current method
        private bool _directXFailedPermanently = false; // Track if DirectX failed with unsupported error
        private bool _notificationShown = false; // Prevent spam notifications

        // Capturing
        public Bitmap? screenCaptureBitmap { get; private set; }
        public Bitmap? directXBitmap { get; private set; }
        private ID3D11Device? _dxDevice;
        private IDXGIOutputDuplication? _deskDuplication;
        private ID3D11Texture2D? _stagingTex;



        // Display change handling
        public readonly object _displayLock = new();
        public bool _displayChangesPending { get; set; } = false;

        // Performance tracking
        private int _consecutiveFailures = 0;
        private const int MAX_CONSECUTIVE_FAILURES = 5;

        // stride matching
        private bool _lastStrideMatch = true;
        private int _lastSrcStride = 0;
        private int _lastDstStride = 0;

        // Last processed desktop presentation timestamp (QPC) - used to reject duplicate frames
        private long _lastPresentTime = 0;

        #endregion
        #region Handlers
        public CaptureManager()
        {
            // Subscribe to display changes FIRST
            DisplayManager.DisplayChanged += OnDisplayChanged;
        }

        private void OnDisplayChanged(object? sender, DisplayChangedEventArgs e)
        {
            lock (_displayLock)
            {
                _displayChangesPending = true;
                _consecutiveFailures = 0;
                DisposeDxgiResources();
            }
            LogManager.Log(LogLevel.Info, "Display change detected. DirectX resources will be reinitialized.");
        }

        public void HandlePendingDisplayChanges()
        {
            lock (_displayLock)
            {
                if (!_displayChangesPending) return;

                try
                {
                    InitializeDxgiDuplication();
                    _displayChangesPending = false;
                }
                catch (Exception ex)
                {

                }
            }
        }

        #endregion
        #region DirectX
        public void InitializeDxgiDuplication()
        {
            DisposeDxgiResources();
            try
            {
                var currentDisplay = DisplayManager.CurrentDisplay;
                if (currentDisplay == null)
                {
                    LogManager.Log(LogLevel.Error, "No current display available. DisplayManager may not be initialized.");
                    throw new InvalidOperationException("No current display available. DisplayManager may not be initialized.");
                }

                using var factory = DXGI.CreateDXGIFactory1<IDXGIFactory1>();
                IDXGIOutput1? targetOutput1 = null;
                IDXGIAdapter1? targetAdapter = null;
                bool foundTarget = false;

                for (uint adapterIndex = 0;
                    factory.EnumAdapters1(adapterIndex, out var adapter).Success;
                    adapterIndex++)
                {
                    LogManager.Log(LogLevel.Info, $"Checking Adapter {adapterIndex}: {adapter.Description.Description.TrimEnd('\0')}");

                    for (uint outputIndex = 0;
                        adapter.EnumOutputs(outputIndex, out var output).Success;
                        outputIndex++)
                    {
                        using (output)
                        {
                            var output1 = output.QueryInterface<IDXGIOutput1>();
                            var outputDesc = output1.Description;
                            var outputBounds = new Vortice.Mathematics.Rect(
                                outputDesc.DesktopCoordinates.Left,
                                outputDesc.DesktopCoordinates.Top,
                                outputDesc.DesktopCoordinates.Right - outputDesc.DesktopCoordinates.Left,
                                outputDesc.DesktopCoordinates.Bottom - outputDesc.DesktopCoordinates.Top);
                            LogManager.Log(LogLevel.Info, $"Found Output {outputIndex}: DeviceName = '{outputDesc.DeviceName.TrimEnd('\0')}', Bounds = {outputBounds}");

                            // Try different matching strategies
                            bool nameMatch = currentDisplay?.DeviceName != null && outputDesc.DeviceName.TrimEnd('\0') == currentDisplay.DeviceName.TrimEnd('\0');
                            bool boundsMatch = currentDisplay?.Bounds != null && outputBounds.Equals(currentDisplay.Bounds);

                            if (nameMatch || boundsMatch)
                            {
                                targetOutput1 = output1;
                                targetAdapter = adapter;
                                foundTarget = true;
                                break;
                            }
                            output1.Dispose();
                        }
                    }

                    if (foundTarget) break;
                }

                // Fallback to specific display index if not found
                if (!foundTarget)
                {
                    int targetIndex = currentDisplay?.Index ?? 0;
                    int currentIndex = 0;

                    for (uint adapterIndex = 0;
                        factory.EnumAdapters1(adapterIndex, out var adapter).Success;
                        adapterIndex++)
                    {
                        for (uint outputIndex = 0;
                            adapter.EnumOutputs(outputIndex, out var output).Success;
                            outputIndex++)
                        {
                            if (currentIndex == targetIndex)
                            {
                                LogManager.Log(LogLevel.Warning, $"Could not match display by name or bounds. Found a fallback index, {targetIndex}.");
                                targetOutput1 = output.QueryInterface<IDXGIOutput1>();
                                targetAdapter = adapter;
                                foundTarget = true;
                                break;
                            }
                            currentIndex++;
                            output.Dispose();
                        }

                        if (foundTarget)
                            break;
                        adapter.Dispose();
                    }
                }

                if (targetAdapter == null || targetOutput1 == null)
                {
                    LogManager.Log(LogLevel.Error, "No suitable display output found for DirectX capture.", true, 6000);
                    throw new Exception("No suitable display output found");
                }

                FeatureLevel[] featureLevels = {
                    FeatureLevel.Level_12_2, // 50 series support
                    FeatureLevel.Level_12_1,
                    FeatureLevel.Level_12_0,
                    FeatureLevel.Level_11_1,
                    FeatureLevel.Level_11_0,
                    FeatureLevel.Level_10_1,
                    FeatureLevel.Level_10_0,
                    FeatureLevel.Level_9_3,
                    FeatureLevel.Level_9_2,
                    FeatureLevel.Level_9_1
                };

                // Create D3D11 device
                var result = D3D11.D3D11CreateDevice(
                    targetAdapter,
                    DriverType.Unknown,
                    DeviceCreationFlags.None,
                    featureLevels,
                    out _dxDevice);

                if (result.Failure || _dxDevice == null)
                {
                    result = D3D11.D3D11CreateDevice(
                      targetAdapter,
                      DriverType.Unknown,
                      DeviceCreationFlags.None,
                      null,
                      out _dxDevice);

                    if (result.Failure || _dxDevice == null)
                    {
                        LogManager.Log(LogLevel.Error, $"Failed to create D3D11 device: {result}", true, 6000);
                        throw new Exception($"Failed to create D3D11 device: {result}");
                    }
                }

                // Create desktop duplication
                _deskDuplication = targetOutput1.DuplicateOutput(_dxDevice);
                _consecutiveFailures = 0; //reset on success

                LogManager.Log(LogLevel.Info, "DirectX Desktop Duplication initialized successfully.");
            }
            catch (SharpGenException ex) when (ex.ResultCode == Vortice.DXGI.ResultCode.Unsupported || ex.HResult == unchecked((int)0x887A0004))
            {
                LogManager.Log(LogLevel.Error, $"DirectX Desktop Duplication not supported on this system: {ex.Message}", true, 6000);
                _directXFailedPermanently = true;
                DisposeDxgiResources();

                Dictionary.dropdownState["Screen Capture Method"] = "GDI+";
                _currentCaptureMethod = "GDI+";

                LogManager.Log(LogLevel.Error, "DirectX Desktop Duplication not supported on this system. Switched to GDI+ capture.", true, 6000);
            }
            catch (Exception ex)
            {
                LogManager.Log(LogLevel.Error, $"Failed to initialize DirectX Desktop Duplication: {ex.Message}", true, 6000);
                DisposeDxgiResources();
                throw;
            }
        }
        private Bitmap? DirectX(Rectangle detectionBox)
        {
            int w = detectionBox.Width;
            int h = detectionBox.Height;
            bool frameAcquired = false;
            IDXGIResource? desktopResource = null;

            Bitmap? resultBitmap = null;

            try
            {
                lock (_displayLock)
                {
                    if (_displayChangesPending)
                    {
                        InitializeDxgiDuplication();
                        _displayChangesPending = false;
                    }
                }

                // Check if we need to reinitialize
                if (_dxDevice == null || _dxDevice.ImmediateContext == null || _deskDuplication == null)
                {
                    InitializeDxgiDuplication();
                    if (_dxDevice == null || _dxDevice.ImmediateContext == null || _deskDuplication == null)
                    {
                        lock (_displayLock) { _displayChangesPending = true; }
                        return null;
                    }
                }

                if (directXBitmap == null || directXBitmap.Width != w || directXBitmap.Height != h)
                {
                    directXBitmap?.Dispose();
                    directXBitmap = new Bitmap(w, h, PixelFormat.Format32bppArgb);
                }

                // Check if we need new staging texture - always match requested size
                if (_stagingTex == null ||
                    _stagingTex.Description.Width != w ||
                    _stagingTex.Description.Height != h)
                {
                    _stagingTex?.Dispose();
                    _stagingTex = _dxDevice.CreateTexture2D(new Texture2DDescription
                    {
                        Width = (uint)w,
                        Height = (uint)h,
                        MipLevels = 1,
                        ArraySize = 1,
                        Format = Format.B8G8R8A8_UNorm,
                        SampleDescription = new(1, 0),
                        Usage = ResourceUsage.Staging,
                        CPUAccessFlags = CpuAccessFlags.Read,
                        BindFlags = BindFlags.None
                    });
                }

                // Wait up to 15ms for the next display presentation (synchronized with game/monitor refresh rate)
                int timeout = _consecutiveFailures > 0 ? 5 : 15;
                var result = _deskDuplication!.AcquireNextFrame((uint)timeout, out var frameInfo, out desktopResource);

                if (result == Vortice.DXGI.ResultCode.WaitTimeout)
                {
                    // No new frame available from display yet - do not reuse stale frame to prevent mouse oscillation
                    _consecutiveFailures = 0;
                    return null;
                }
                else if (result == Result.Ok && frameInfo.AccumulatedFrames == 0)
                {
                    // Desktop image was not updated (only mouse cursor or metadata)
                    desktopResource?.Dispose();
                    try { _deskDuplication.ReleaseFrame(); } catch { }
                    return null;
                }
                else if (result == Result.Ok && frameInfo.LastPresentTime != 0 && frameInfo.LastPresentTime == _lastPresentTime)
                {
                    // The OS handed back the same desktop image we already processed. Running the aim
                    // loop twice on one frame double-applies the correction, which is exactly the kind
                    // of repeated push that turns a proportional aim loop into an overshoot oscillation.
                    // (AccumulatedFrames already catches most of this; this catches the rest.)
                    desktopResource?.Dispose();
                    try { _deskDuplication.ReleaseFrame(); } catch { }
                    return null;
                }
                else if (result == Vortice.DXGI.ResultCode.DeviceRemoved || result == Vortice.DXGI.ResultCode.AccessLost)
                {
                    // Device lost - flag for reinitialization
                    _consecutiveFailures++;
                    if (_consecutiveFailures >= MAX_CONSECUTIVE_FAILURES)
                        lock (_displayLock) { _displayChangesPending = true; }

                    return null;
                }
                else if (result != Result.Ok)
                {
                    _consecutiveFailures++;
                    return null;
                }

                frameAcquired = true;
                _consecutiveFailures = 0; // Reset on successful acquisition
                _lastPresentTime = frameInfo.LastPresentTime;

                using (var screenTexture = desktopResource.QueryInterface<ID3D11Texture2D>())
                {
                    #region Display Bounds
                    var desc = screenTexture.Description;
                    int texWidth = (int)desc.Width;
                    int texHeight = (int)desc.Height;

                    // Convert absolute screen coordinates to display-relative coordinates
                    int relativeDetectionLeft = detectionBox.Left - DisplayManager.ScreenLeft;
                    int relativeDetectionTop = detectionBox.Top - DisplayManager.ScreenTop;
                    int relativeDetectionRight = relativeDetectionLeft + detectionBox.Width;
                    int relativeDetectionBottom = relativeDetectionTop + detectionBox.Height;

                    // Calculate the visible portion in display-relative coordinates clamped to texture dimensions
                    int srcLeft = Math.Clamp(relativeDetectionLeft, 0, texWidth);
                    int srcTop = Math.Clamp(relativeDetectionTop, 0, texHeight);
                    int srcRight = Math.Clamp(relativeDetectionRight, 0, texWidth);
                    int srcBottom = Math.Clamp(relativeDetectionBottom, 0, texHeight);

                    // Only copy if there's a visible region
                    if (srcRight > srcLeft && srcBottom > srcTop)
                    {
                        var box = new Box(srcLeft, srcTop, 0, srcRight, srcBottom, 1);

                        _dxDevice.ImmediateContext.CopySubresourceRegion(
                               _stagingTex, 0,
                               (uint)(srcLeft - relativeDetectionLeft),
                               (uint)(srcTop - relativeDetectionTop),
                               0,
                               screenTexture, 0, box);
                    }
                    else
                    {
                        return null;
                    }

                    #endregion

                    // Release desktop resource as early as possible so DXGI doesn't block the next frame present
                    desktopResource?.Dispose();
                    desktopResource = null;
                    try
                    {
                        _deskDuplication.ReleaseFrame();
                        frameAcquired = false;
                    }
                    catch { }

                    #region Bitmap
                    var map = _dxDevice.ImmediateContext.Map(_stagingTex, 0, MapMode.Read, Vortice.Direct3D11.MapFlags.None);
                    var boundsRect = new Rectangle(0, 0, w, h);
                    BitmapData? mapDest = directXBitmap.LockBits(boundsRect, ImageLockMode.WriteOnly, directXBitmap.PixelFormat);

                    try
                    {
                        unsafe
                        {
                            byte* src = (byte*)map.DataPointer;
                            byte* dst = (byte*)mapDest.Scan0;
                            int srcStride = (int)map.RowPitch;
                            int dstStride = mapDest.Stride;

                            int copyBytesPerRow = Math.Min(srcStride, dstStride);
                            for (int y = 0; y < h; y++)
                            {
                                Buffer.MemoryCopy(src, dst, dstStride, copyBytesPerRow);
                                src += srcStride;
                                dst += dstStride;
                            }

                            if (Dictionary.toggleState["Third Person Support"])
                            {
                                int width = w / 2;
                                int height = h / 2;
                                int startY = h - height;

                                byte* basePtr = (byte*)mapDest.Scan0;
                                for (int y = startY; y < h; y++)
                                {
                                    byte* rowPtr = basePtr + (y * dstStride);
                                    for (int x = 0; x < width; x++)
                                    {
                                        int pixelOffset = x * 4;
                                        rowPtr[pixelOffset + 0] = 0;
                                        rowPtr[pixelOffset + 1] = 0;
                                        rowPtr[pixelOffset + 2] = 0;
                                        rowPtr[pixelOffset + 3] = 255;
                                    }
                                }
                            }
                        }
                    }
                    finally
                    {
                        directXBitmap.UnlockBits(mapDest);
                        _dxDevice.ImmediateContext.Unmap(_stagingTex, 0);
                    }
                    #endregion

                    resultBitmap = (Bitmap)directXBitmap.Clone();
                    return resultBitmap;
                }
            }
            catch (Exception e)
            {
                LogManager.Log(LogLevel.Error, $"DirectX capture error: {e.Message}");

                if (++_consecutiveFailures >= MAX_CONSECUTIVE_FAILURES)
                    lock (_displayLock) { _displayChangesPending = true; }

                return null;
            }
            finally
            {
                desktopResource?.Dispose();
                try
                {
                    if (frameAcquired && _deskDuplication != null)
                    {
                        _deskDuplication.ReleaseFrame();
                    }
                }
                catch { }
            }
        }
        #endregion

        #region GDI
        public Bitmap GDIScreen(Rectangle detectionBox)
        {
            if (_dxDevice != null || _deskDuplication != null)
            {
                DisposeDxgiResources();
            }

            if (screenCaptureBitmap == null || screenCaptureBitmap.Width != detectionBox.Width || screenCaptureBitmap.Height != detectionBox.Height)
            {
                screenCaptureBitmap?.Dispose();
                screenCaptureBitmap = new Bitmap(detectionBox.Width, detectionBox.Height, PixelFormat.Format32bppArgb);
            }

            try
            {
                using (var g = Graphics.FromImage(screenCaptureBitmap))
                {
                    g.CopyFromScreen(
                        detectionBox.Left,
                        detectionBox.Top,
                        0, 0,
                        detectionBox.Size,
                        CopyPixelOperation.SourceCopy
                    );

                    if (Dictionary.toggleState["Third Person Support"])
                    {
                        int width = screenCaptureBitmap.Width / 2;
                        int height = screenCaptureBitmap.Height / 2;
                        int startY = screenCaptureBitmap.Height - height;

                        using var brush = new SolidBrush(System.Drawing.Color.Black);
                        g.FillRectangle(brush, 0, startY, width, height);
                    }
                }

                // Clone the bitmap to avoid race conditions with Sticky Aim / SaveFrame
                // The source bitmap is reused, so returning it directly can cause crashes
                // if the caller is still using it when the next frame capture starts
                return (Bitmap)screenCaptureBitmap.Clone();
            }
            catch (Exception ex)
            {
                LogManager.Log(LogLevel.Error, $"GDI+ screen capture failed: {ex.Message}");
                throw;
            }
        }
        #endregion

        public Bitmap? ScreenGrab(Rectangle detectionBox)
        {
            string selectedMethod = Dictionary.dropdownState["Screen Capture Method"];

            // If DirectX failed permanently, force GDI+
            if (_directXFailedPermanently && selectedMethod == "DirectX")
            {
                Dictionary.dropdownState["Screen Capture Method"] = "GDI+";
                selectedMethod = "GDI+";
                _currentCaptureMethod = "GDI+";
            }

            // Handle method switch
            if (selectedMethod != _currentCaptureMethod)
            {
                // Dispose bitmap when switching methods
                screenCaptureBitmap?.Dispose();
                screenCaptureBitmap = null;

                directXBitmap?.Dispose();
                directXBitmap = null;

                _currentCaptureMethod = selectedMethod;
                _notificationShown = false; // Reset notification flag on method change

                // Dispose DX resources when switching to GDI
                if (selectedMethod == "GDI+")
                {
                    DisposeDxgiResources();
                }
                else
                {
                    InitializeDxgiDuplication();
                }
            }

            if (selectedMethod == "DirectX" && !_directXFailedPermanently)
            {
                return DirectX(detectionBox);
            }
            else
            {
                return GDIScreen(detectionBox);
            }
        }

        #region dispose
        public void DisposeDxgiResources()
        {
            lock (_displayLock)
            {
                try
                {

                    // Try to release any pending frame
                    if (_deskDuplication != null)
                    {
                        try
                        {
                            _deskDuplication.ReleaseFrame();
                        }
                        catch { }
                    }

                    _deskDuplication?.Dispose();
                    _stagingTex?.Dispose();
                    _dxDevice?.Dispose();
                    directXBitmap?.Dispose();

                    _deskDuplication = null;
                    _stagingTex = null;
                    _dxDevice = null;
                    _lastPresentTime = 0;

                    // Small delay to ensure resources are fully released
                    //System.Threading.Thread.Sleep(50);
                }
                catch (Exception ex)
                {
                    LogManager.Log(LogLevel.Error, $"Error disposing DXGI resources: {ex.Message}");
                }
            }
        }
        public void Dispose()
        {
            DisplayManager.DisplayChanged -= OnDisplayChanged;
            DisposeDxgiResources();
            screenCaptureBitmap?.Dispose();
        }
        #endregion
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