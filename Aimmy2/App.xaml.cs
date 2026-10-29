using Aimmy2.Theme;
using Class;
using System.Windows;

namespace Aimmy2
{
    public partial class App : Application
    {
        [System.Runtime.InteropServices.DllImport("kernel32.dll", CharSet = System.Runtime.InteropServices.CharSet.Auto, SetLastError = true)]
        private static extern bool SetDllDirectory(string lpPathName);

        protected override void OnStartup(StartupEventArgs e)
        {
            try
            {
                string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                string binDir = System.IO.Path.Combine(baseDir, "bin");
                string trtCache = System.IO.Path.Combine(binDir, "tensorrt_cache");
                System.IO.Directory.CreateDirectory(trtCache);

                SetDllDirectory(baseDir);

                string currentPath = Environment.GetEnvironmentVariable("PATH") ?? "";
                if (!currentPath.Contains(baseDir))
                {
                    Environment.SetEnvironmentVariable("PATH", $"{baseDir};{binDir};{currentPath}", EnvironmentVariableTarget.Process);
                }
            }
            catch { }

            // Initialize the application theme from saved settings
            InitializeTheme();

            // Set shutdown mode to prevent app from closing when startup window closes
            ShutdownMode = ShutdownMode.OnExplicitShutdown;

#if DEBUG
            var _mainWindow = new MainWindow();
            MainWindow = _mainWindow;
            _mainWindow.Show();
            return;
#endif
            // code IS reachable, only in release though
            try
            {
                // Create and show startup window
                var startupWindow = new StartupWindow();
                startupWindow.Show();

                // Reset shutdown mode after startup window is shown
                ShutdownMode = ShutdownMode.OnMainWindowClose;
            }
            catch (Exception ex)
            {
                // If startup window fails, launch main window directly
                MessageBox.Show($"Startup animation failed: {ex.Message}\nLaunching main application...",
                              "AntiGravity", MessageBoxButton.OK, MessageBoxImage.Information);

                var mainWindow = new MainWindow();
                MainWindow = mainWindow;
                mainWindow.Show();

                ShutdownMode = ShutdownMode.OnMainWindowClose;
            }
        }

        private void InitializeTheme()
        {
            try
            {
                // Load the color state configuration
                var colorState = new Dictionary<string, dynamic>
                {
                    { "Theme Color", "#FF722ED1" }
                };

                // Load saved colors
                SaveDictionary.LoadJSON(colorState, "bin\\colors.cfg");

                // Apply theme color if found
                if (colorState.TryGetValue("Theme Color", out var themeColor) && themeColor is string colorString)
                {
                    ThemeManager.SetThemeColor(colorString);
                }
                else
                {
                    // Use default purple if no saved color
                    ThemeManager.SetThemeColor("#FF722ED1");
                }
            }
            catch (Exception ex)
            {
                // Log error and use default color
                ThemeManager.SetThemeColor("#FF722ED1");
            }
        }
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