
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace CudaKernelLauncher
{
	public class CudaContextHandling
	{
		// ----- ATTRIBUTES ----- \\
		public string Repopath;

		public int DeviceId = 0;

		public PrimaryContext? Ctx = null;
		


		public int LogInterval = 100;

		// ----- OBJECTS ----- \\
		public ListBox LogBox;
		private ComboBox DevicesCombo;
		private Label VramLabel;
		private ProgressBar VramBar;

		public CudaMemoryHandling? MemH = null;
		public CudaFftHandling? FftH = null;
		public CudaKernelHandling? KernelH = null;



		// ----- LAMBDA ----- \\
		public CUdevice[] Devices => GetDevices();




		// ----- CONSTRUCTOR ----- \\
		public CudaContextHandling(string repopath, ListBox? listBox_log = null, ComboBox? comboBox_devices = null, Label? label_vram = null, ProgressBar? progressBar_vram = null)
		{
			// Set attributes
			this.Repopath = repopath;
			this.LogBox = listBox_log ?? new ListBox();
			this.DevicesCombo = comboBox_devices ?? new ComboBox();
			this.VramLabel = label_vram ?? new Label();
			this.VramBar = progressBar_vram ?? new ProgressBar();

			// Register events
			DevicesCombo.SelectedIndexChanged += (s, e) => InitContext(DevicesCombo.SelectedIndex);

			// Fill devices combo
			FillDevicesCombo();

		}





		// ----- METHODS ----- \\
		// Log
		public void Log(string message, string inner = "", int layer = 1, bool update = false)
		{
			string msg = "[" + DateTime.Now.ToString("HH:mm:ss.fff") + "] ";
			msg += "<CUDA>";

			for (int i = 0; i <= layer; i++)
			{
				msg += " - ";
			}

			msg += message;

			if (inner != "")
			{
				msg += "  (" + inner + ")";
			}

			if (update)
			{
				LogBox.Items[LogBox.Items.Count - 1] = msg;
			}
			else
			{
				LogBox.Items.Add(msg);
				LogBox.SelectedIndex = LogBox.Items.Count - 1;
			}
		}

		// Pre Context & info
		public int GetDeviceCount()
		{
			return CudaContext.GetDeviceCount();
		}

		private CUdevice[] GetDevices()
		{
			int devCount = GetDeviceCount();
			if (devCount == 0)
			{
				Log("No CUDA devices found.");
				return [];
			}

			CUdevice[] devices = new CUdevice[devCount];
			for (int i = 0; i < devCount; i++)
			{
				devices[i] = new CUdevice(i);
			}

			return devices;
		}

		public void FillDevicesCombo()
		{
			DevicesCombo.Items.Clear();
			for (int i = 0; i < Devices.Length; i++)
			{
				string name = CudaContext.GetDeviceName(i);

				DevicesCombo.Items.Add(name);
			}

			// Add entry CPU only
			DevicesCombo.Items.Add(" - CPU only - ");

			DevicesCombo.SelectedIndex = Math.Min(DeviceId, Devices.Length - 1);
		}

		// Context & Dispose
		public void InitContext(int deviceId)
		{
			// Dipose previous context
			Dispose();

			// Check if device is valid
			if (deviceId < 0 || deviceId >= Devices.Length)
			{
				DeviceId = -1;
				Log("No device selected", "Id: " + deviceId, 1);
				return;
			}

			// Set device id
			DeviceId = deviceId;

			// Create context
			Ctx = new PrimaryContext(deviceId);
			Ctx.SetCurrent();

			// Create memory handling & fft handling
			MemH = new CudaMemoryHandling(this, Ctx);
			FftH = new CudaFftHandling(this, Ctx);
			KernelH = new CudaKernelHandling(this, Ctx);

			// Log
			Log("Context initialized on device " + "Id: " + deviceId);
		}

		public void Dispose()
		{
			// Dispose context
			Ctx?.Dispose();
			Ctx = null;

			// Dispose memory handling
			MemH?.Dispose();

			// Dispose fft handling
			FftH?.Dispose();
		}

		// VRAM info
		public int[] GetVramInfo()
		{
			// Abort if no device selected
			if (Ctx == null)
			{
				VramLabel.Text = "No device selected";
				VramBar.Value = 0;
				return [0, 0, 0];
			}

			// Get device memory info
			long[] usage = [0, 0, 0];
			try
			{
				usage[0] = Ctx.GetTotalDeviceMemorySize() / 1024 / 1024;
				usage[1] = Ctx.GetFreeDeviceMemorySize() / 1024 / 1024;
				usage[2] = usage[0] - usage[1];
			}
			catch (Exception e)
			{
				Log("Failed to get VRAM info", e.Message, 1);
			}

			// Update UI
			VramLabel.Text = $"VRAM: {usage[2]} MB / {usage[0]} MB";
			VramBar.Maximum = (int) usage[0];
			VramBar.Value = (int) usage[2];

			// Return info
			return [(int) usage[0], (int) usage[1], (int) usage[2]];
		}

		public void FillKernelsListbox(ListBox listBox_kernels)
		{
			listBox_kernels.Items.Clear();

			// Fill with every file name in the kernels folder (PTX)
			string kernelsPath = Path.Combine(Repopath, "Kernels\\PTX");
			string[] files = Directory.GetFiles(kernelsPath, "*.ptx");
			foreach ( string file in files )
			{
				listBox_kernels.Items.Add(Path.GetFileNameWithoutExtension(file));
			}

			// Select no item
			listBox_kernels.SelectedIndex = -1;
		}

		public void CompileAllKernels()
		{
			// Check KernelH
			if (KernelH == null)
			{
				Log("Kernel handling not initialized", "Id: " + DeviceId, 1);
				return;
			}

			// Compile all .cu files with KernelH
			string[] cuPaths = Directory.GetFiles(Path.Combine(Repopath, "Kernels\\CU"), "*.cu");
			foreach (string cuPath in cuPaths)
			{
				KernelH.CompileKernel(cuPath);
			}
		}

		
	}
}