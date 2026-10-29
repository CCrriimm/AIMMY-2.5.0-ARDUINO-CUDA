using Other;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Aimmy2.Other
{
    public static class PerformanceOptimizer
    {
        public enum GpuPriorityClass
        {
            Idle = 0,
            BelowNormal = 1,
            Normal = 2,
            AboveNormal = 3,
            High = 4,
            Realtime = 5
        }

        #region P/Invoke Signatures

        [DllImport("gdi32.dll", EntryPoint = "D3DKMTSetProcessSchedulingPriorityClass")]
        private static extern int D3DKMTSetProcessSchedulingPriorityClass(IntPtr hProcess, int priority);

        [DllImport("gdi32.dll", EntryPoint = "D3DKMTGetProcessSchedulingPriorityClass")]
        private static extern int D3DKMTGetProcessSchedulingPriorityClass(IntPtr hProcess, out int priority);

        [DllImport("kernel32.dll")]
        private static extern IntPtr GetCurrentProcess();

        [StructLayout(LayoutKind.Sequential)]
        private struct PROCESS_POWER_THROTTLING_STATE
        {
            public uint Version;
            public uint ControlMask;
            public uint StateMask;
        }

        private const int ProcessPowerThrottling = 4;
        private const uint PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1;
        private const uint PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1;

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool SetProcessInformation(
            IntPtr hProcess,
            int processInformationClass,
            ref PROCESS_POWER_THROTTLING_STATE processInformation,
            uint processInformationSize);

        [DllImport("winmm.dll", EntryPoint = "timeBeginPeriod")]
        private static extern uint timeBeginPeriod(uint uPeriod);

        [DllImport("winmm.dll", EntryPoint = "timeEndPeriod")]
        private static extern uint timeEndPeriod(uint uPeriod);

        [StructLayout(LayoutKind.Sequential)]
        private struct LUID
        {
            public uint LowPart;
            public int HighPart;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct LUID_AND_ATTRIBUTES
        {
            public LUID Luid;
            public uint Attributes;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct TOKEN_PRIVILEGES
        {
            public uint PrivilegeCount;
            public LUID_AND_ATTRIBUTES Privileges;
        }

        private const uint TOKEN_ADJUST_PRIVILEGES = 0x0020;
        private const uint TOKEN_QUERY = 0x0008;
        private const uint SE_PRIVILEGE_ENABLED = 0x00000002;
        private const string SE_INC_BASE_PRIORITY_NAME = "SeIncreaseBasePriorityPrivilege";

        [DllImport("advapi32.dll", SetLastError = true)]
        private static extern bool OpenProcessToken(IntPtr ProcessHandle, uint DesiredAccess, out IntPtr TokenHandle);

        [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Auto)]
        private static extern bool LookupPrivilegeValue(string? lpSystemName, string lpName, out LUID lpLuid);

        [DllImport("advapi32.dll", SetLastError = true)]
        private static extern bool AdjustTokenPrivileges(IntPtr TokenHandle, bool DisableAllPrivileges, ref TOKEN_PRIVILEGES NewState, int BufferLength, IntPtr PreviousState, IntPtr ReturnLength);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool CloseHandle(IntPtr hObject);

        #endregion

        private static bool _timerInitialized = false;

        public static void InitializeTimer()
        {
            if (!_timerInitialized)
            {
                try
                {
                    timeBeginPeriod(1);
                    _timerInitialized = true;
                }
                catch { }
            }
        }

        private static bool EnableIncreaseBasePriorityPrivilege()
        {
            try
            {
                if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, out IntPtr hToken))
                    return false;

                try
                {
                    if (!LookupPrivilegeValue(null, SE_INC_BASE_PRIORITY_NAME, out LUID luid))
                        return false;

                    TOKEN_PRIVILEGES tp = new TOKEN_PRIVILEGES
                    {
                        PrivilegeCount = 1,
                        Privileges = new LUID_AND_ATTRIBUTES
                        {
                            Luid = luid,
                            Attributes = SE_PRIVILEGE_ENABLED
                        }
                    };

                    return AdjustTokenPrivileges(hToken, false, ref tp, 0, IntPtr.Zero, IntPtr.Zero);
                }
                finally
                {
                    CloseHandle(hToken);
                }
            }
            catch
            {
                return false;
            }
        }

        public static bool DisablePowerThrottling()
        {
            try
            {
                var state = new PROCESS_POWER_THROTTLING_STATE
                {
                    Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION,
                    ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
                    StateMask = 0 // 0 = Disable power throttling (run at full performance in background)
                };

                bool result = SetProcessInformation(
                    GetCurrentProcess(),
                    ProcessPowerThrottling,
                    ref state,
                    (uint)Marshal.SizeOf(state));

                if (result)
                {
                    LogManager.Log(LogManager.LogLevel.Info, "Windows Power Throttling / EcoQoS disabled for AntiGravity.", false);
                }
                return result;
            }
            catch (Exception ex)
            {
                LogManager.Log(LogManager.LogLevel.Warning, $"Failed to configure Power Throttling: {ex.Message}", false);
                return false;
            }
        }

        public static bool SetGpuPriority(GpuPriorityClass priority)
        {
            try
            {
                if (priority == GpuPriorityClass.Realtime)
                {
                    EnableIncreaseBasePriorityPrivilege();
                }

                int status = D3DKMTSetProcessSchedulingPriorityClass(GetCurrentProcess(), (int)priority);
                if (status == 0) // STATUS_SUCCESS
                {
                    LogManager.Log(LogManager.LogLevel.Info, $"GPU Scheduling Priority set to {priority}.", true, 2000);
                    return true;
                }

                // If Realtime failed, fallback to High
                if (priority == GpuPriorityClass.Realtime)
                {
                    status = D3DKMTSetProcessSchedulingPriorityClass(GetCurrentProcess(), (int)GpuPriorityClass.High);
                    if (status == 0)
                    {
                        LogManager.Log(LogManager.LogLevel.Info, "GPU Scheduling Priority set to High (Realtime not permitted).", true, 2000);
                        return true;
                    }
                }

                LogManager.Log(LogManager.LogLevel.Warning, $"Failed to set GPU Scheduling Priority (Status code: 0x{status:X8}).", true, 3000);
                return false;
            }
            catch (Exception ex)
            {
                LogManager.Log(LogManager.LogLevel.Warning, $"D3DKMT call failed: {ex.Message}", false);
                return false;
            }
        }

        public static GpuPriorityClass GetCurrentGpuPriority()
        {
            try
            {
                int status = D3DKMTGetProcessSchedulingPriorityClass(GetCurrentProcess(), out int prio);
                if (status == 0 && Enum.IsDefined(typeof(GpuPriorityClass), prio))
                {
                    return (GpuPriorityClass)prio;
                }
            }
            catch { }
            return GpuPriorityClass.Normal;
        }

        public static void SetProcessCpuPriority(ProcessPriorityClass priorityClass)
        {
            try
            {
                using var currentProc = Process.GetCurrentProcess();
                if (currentProc.PriorityClass != priorityClass)
                {
                    currentProc.PriorityClass = priorityClass;
                    LogManager.Log(LogManager.LogLevel.Info, $"CPU Process Priority set to {priorityClass}.", false);
                }
            }
            catch (Exception ex)
            {
                LogManager.Log(LogManager.LogLevel.Warning, $"Failed to set Process CPU Priority: {ex.Message}", false);
            }
        }

        public static void OptimizeAll(bool enableRealtimeGpu = true)
        {
            InitializeTimer();
            DisablePowerThrottling();
            SetProcessCpuPriority(ProcessPriorityClass.High);

            if (enableRealtimeGpu)
            {
                SetGpuPriority(GpuPriorityClass.Realtime);
            }
            else
            {
                SetGpuPriority(GpuPriorityClass.Normal);
            }
        }
    }
}
