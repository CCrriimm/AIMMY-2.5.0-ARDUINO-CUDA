using System;
using System.IO.Ports;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows;

namespace MouseMovementLibraries.MakcuSupport
{
    internal class MakcuInput
    {
        private SerialPort? serialPort;
        private readonly object _writeLock = new();

        public string ComPort { get; private set; }
        public int BaudRate { get; private set; }

        public bool IsOpen => serialPort != null && serialPort.IsOpen;

        public MakcuInput(string comPortNumber, int baudRate = 4000000)
        {
            string portNumber = Regex.Match(comPortNumber ?? "", @"\d+").Value;

            if (string.IsNullOrEmpty(portNumber))
                throw new ArgumentException("Invalid COM port input.");

            ComPort = $"COM{portNumber}";
            BaudRate = baudRate;

            OpenPort(ComPort, BaudRate);
        }

        private void OpenPort(string portName, int baud)
        {
            try
            {
                serialPort = new SerialPort(portName, baud)
                {
                    Encoding = Encoding.ASCII,
                    NewLine = "\r\n",
                    DtrEnable = true,
                    RtsEnable = true,
                    WriteBufferSize = 4096,
                    ReadBufferSize = 4096,
                    WriteTimeout = 500,
                    ReadTimeout = 500
                };

                serialPort.Open();
            }
            catch (Exception ex)
            {
                // If 4M baud fails to open on certain serial drivers, try 115200 as fallback
                if (baud != 115200)
                {
                    try
                    {
                        serialPort?.Dispose();
                        serialPort = new SerialPort(portName, 115200)
                        {
                            Encoding = Encoding.ASCII,
                            NewLine = "\r\n",
                            DtrEnable = true,
                            RtsEnable = true,
                            WriteBufferSize = 4096,
                            ReadBufferSize = 4096,
                            WriteTimeout = 500,
                            ReadTimeout = 500
                        };
                        serialPort.Open();
                        BaudRate = 115200;
                        return;
                    }
                    catch { }
                }

                MessageBox.Show($"Failed to open Makcu on {portName} at {baud} baud:\n{ex.Message}", "Makcu COM Port Error", MessageBoxButton.OK, MessageBoxImage.Error);
                throw;
            }
        }

        public void Move(int x, int y)
        {
            if (serialPort == null || !serialPort.IsOpen)
                return;

            string cmd = $"km.move({x},{y})\r\n";
            WriteCommand(cmd);
        }

        public void SendClick(int click)
        {
            if (serialPort == null || !serialPort.IsOpen)
                return;

            // 1 = down/pressed, 0 = up/released
            string cmd = click == 1 ? "km.left(1)\r\n" : "km.left(0)\r\n";
            WriteCommand(cmd);
        }

        public void SendRightClick(int click)
        {
            if (serialPort == null || !serialPort.IsOpen)
                return;

            string cmd = click == 1 ? "km.right(1)\r\n" : "km.right(0)\r\n";
            WriteCommand(cmd);
        }

        public void SendMiddleClick(int click)
        {
            if (serialPort == null || !serialPort.IsOpen)
                return;

            string cmd = click == 1 ? "km.middle(1)\r\n" : "km.middle(0)\r\n";
            WriteCommand(cmd);
        }

        public void SendWheel(int delta)
        {
            if (serialPort == null || !serialPort.IsOpen)
                return;

            string cmd = $"km.wheel({delta})\r\n";
            WriteCommand(cmd);
        }

        private void WriteCommand(string command)
        {
            try
            {
                lock (_writeLock)
                {
                    if (serialPort != null && serialPort.IsOpen)
                    {
                        byte[] bytes = Encoding.ASCII.GetBytes(command);
                        serialPort.BaseStream.Write(bytes, 0, bytes.Length);
                    }
                }
            }
            catch { }
        }

        public void Close()
        {
            try
            {
                lock (_writeLock)
                {
                    if (serialPort != null && serialPort.IsOpen)
                    {
                        serialPort.Close();
                    }
                    serialPort?.Dispose();
                    serialPort = null;
                }
            }
            catch { }
        }
    }
}
