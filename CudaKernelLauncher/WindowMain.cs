namespace CudaKernelLauncher
{
	public partial class WindowMain : Form
	{
		// ----- ATTRIBUTES ----- \\
		public string Repopath;



		private int _lastChunkSize = 256;

		// ----- OBJECTS ----- \\
		public GuiBuilder GuiB;
		public AudioHandling AH;
		public CudaContextHandling CCH;





		// ----- CONSTRUCTOR ----- \\
		public WindowMain()
		{
			InitializeComponent();

			// Set repopath
			Repopath = GetRepopath(true);

			// Init objects
			GuiB = new GuiBuilder(this);
			AH = new AudioHandling(listBox_tracks, pictureBox_wave, button_play, hScrollBar_offset);
			CCH = new CudaContextHandling(Repopath, listBox_log, comboBox_devices, label_vram, progressBar_vram);

			// Register events
			numericUpDown_chunkSize.Click += (s, e) => ToggleChunking();
			panel_toggleChunking.Click += (s, e) => ToggleChunking();

			// Import resources
			ImportResources();

			// Start GUI
			RegisterTooltips();
			ToggleUI();

		}





		// ----- METHODS ----- \\
		public string GetRepopath(bool root = false)
		{
			string repo = AppDomain.CurrentDomain.BaseDirectory;

			if (root)
			{
				repo += @"..\..\..\";
			}

			repo = Path.GetFullPath(repo);

			return repo;
		}

		private void RegisterTooltips()
		{
			// Create & register tooltip for numeric & panel chunking
			ToolTip chuckingNumericToolTip = new ToolTip();
			chuckingNumericToolTip.SetToolTip(numericUpDown_chunkSize, "CTRL-click to toggle chunking");
		}

		public void ImportResources()
		{
			// In Repopath\Resources, add every file (audio)
			string path = Repopath + @"Resources\";
			string[] files = Directory.GetFiles(path);
			foreach (string file in files)
			{
				AH.AddTrack(file);
			}

			// Toggle UI
			ToggleUI();
		}

		public void MoveTrack()
		{
			// Abort if no track or memh
			if (AH.CurrentTrack == null || CCH.MemH == null)
			{
				return;
			}

			// Move track between Host <--> Device
			if (AH.CurrentTrack.OnHost)
			{
				if (numericUpDown_chunkSize.Enabled)
				{
					// Make chunks
					var chunks = AH.CurrentTrack.MakeChunks((int) numericUpDown_chunkSize.Value);

					// Push chunks to device
					AH.CurrentTrack.Pointer = CCH.MemH.PushData(chunks);
				}
				else
				{
					// Move to Device
					AH.CurrentTrack.Pointer = CCH.MemH.PushData([AH.CurrentTrack.Data]);
				}

				// Void host data
				AH.CurrentTrack.Data = [];
			}
			else if (AH.CurrentTrack.OnDevice)
			{
				if (numericUpDown_chunkSize.Enabled)
				{
					// Pull chunks from device
					var chunks = CCH.MemH.PullData<float>(AH.CurrentTrack.Pointer);

					// Merge chunks
					AH.CurrentTrack.AggregateChunks(chunks);
				}
				else
				{
					// Pull data from device
					AH.CurrentTrack.Data = CCH.MemH.PullData<float>(AH.CurrentTrack.Pointer).FirstOrDefault() ?? [];
				}

				// Void pointer
				AH.CurrentTrack.Pointer = 0;
			}

			// Toggle UI
			ToggleUI();

		}

		public void FourierTransform()
		{
			// Abort if no track or fft
			if (AH.CurrentTrack == null || CCH.FftH == null)
			{
				return;
			}

			// Verify track on device
			if (!AH.CurrentTrack.OnDevice)
			{
				// Move track to device
				MoveTrack();
			}

			// FFT or IFFT depending on form of track
			if (AH.CurrentTrack.Form == 'f')
			{
				// FFT -> Pointer
				AH.CurrentTrack.Pointer = CCH.FftH.PerformFFT(AH.CurrentTrack.Pointer);
				AH.CurrentTrack.Form = 'c';
			}
			else if (AH.CurrentTrack.Form == 'c')
			{
				// IFFT -> Pointer
				AH.CurrentTrack.Pointer = CCH.FftH.PerformIFFT(AH.CurrentTrack.Pointer);
				AH.CurrentTrack.Form = 'f';
			}

			// Toggle UI
			ToggleUI();
		}

		public void ToggleChunking()
		{
			// Require CTRL down & on host
			if (ModifierKeys != Keys.Control || (AH.CurrentTrack != null && !AH.CurrentTrack.OnHost))
			{
				return;
			}

			// Toggle numeric 
			numericUpDown_chunkSize.Enabled = !numericUpDown_chunkSize.Enabled;
		}

		public void ToggleUI()
		{
			// PictureBox wave
			pictureBox_wave.Image = AH.CurrentWave;

			// Set currently loaded kernel label
			label_kernelLoaded.Text = CCH.KernelH?.KernelName ?? "None loaded";
			label_kernelLoaded.ForeColor = CCH.KernelH?.KernelName != null ? Color.DarkGreen : Color.Black;

			// Fill kernels
			CCH.FillKernelsListbox(listBox_kernels);

			// Toggle play button
			button_play.Enabled = AH.CurrentTrack != null && AH.CurrentTrack.OnHost;

			// Toggle export button
			button_export.Enabled = AH.CurrentTrack != null && AH.CurrentTrack.OnHost;

			// Toggle move button
			button_move.Enabled = AH.CurrentTrack != null && (AH.CurrentTrack.OnHost || AH.CurrentTrack.OnDevice);
			button_move.Text = AH.CurrentTrack != null && AH.CurrentTrack.OnHost ? "-> DEV" : "Host <-";

			// Toggle fourier button
			button_fourier.Enabled = AH.CurrentTrack != null && (AH.CurrentTrack.OnHost || AH.CurrentTrack.OnDevice) && CCH.FftH != null;
			button_fourier.Text = AH.CurrentTrack != null && AH.CurrentTrack.Form == 'f' ? "FFT" : "I-FFT";

			// Toggle normalize button
			button_normalize.Enabled = AH.CurrentTrack != null && (AH.CurrentTrack.OnHost || AH.CurrentTrack.OnDevice);

			// Toggle compile button
			button_compileAll.Enabled = CCH.KernelH != null;

			// Toggle execute button
			button_kernelExecute.Enabled = CCH.KernelH != null && CCH.KernelH.Kernel != null;

		}




		// ----- EVENTS ----- \\
		private void button_import_Click(object sender, EventArgs e)
		{
			// OFD at MyMusic for .wav, .mp3, .flac
			OpenFileDialog ofd = new OpenFileDialog();
			ofd.Title = "Import audio";
			ofd.Filter = "Audio Files (*.wav, *.mp3, *.flac)|*.wav;*.mp3;*.flac";
			ofd.InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyMusic);
			ofd.Multiselect = false;

			// OFD show
			if (ofd.ShowDialog() == DialogResult.OK)
			{
				// Import
				AH.AddTrack(ofd.FileName);
			}

			// Toggle UI
			ToggleUI();
		}

		private void button_export_Click(object sender, EventArgs e)
		{
			// SFD at MyMusic for .wav
			SaveFileDialog sfd = new SaveFileDialog();
			sfd.Title = "Export audio";
			sfd.FileName = AH.CurrentTrack?.Name ?? "audio.wav";
			sfd.Filter = "Wave File (*.wav)|*.wav";
			sfd.InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyMusic);

			if (sfd.ShowDialog() == DialogResult.OK)
			{
				// Export
				AH.CurrentTrack?.ExportAudioWav(Repopath);
			}

			// Toggle UI
			ToggleUI();
		}

		private void listBox_tracks_SelectedIndexChanged(object sender, EventArgs e)
		{
			ToggleUI();
		}

		private void button_move_Click(object sender, EventArgs e)
		{
			MoveTrack();
		}

		private void numericUpDown_chunkSize_ValueChanged(object sender, EventArgs e)
		{
			// Double if inc, halve if dec
			if (numericUpDown_chunkSize.Value > _lastChunkSize)
			{
				numericUpDown_chunkSize.Value = Math.Min(numericUpDown_chunkSize.Maximum, _lastChunkSize * 2);
			}
			else if (numericUpDown_chunkSize.Value < _lastChunkSize)
			{
				numericUpDown_chunkSize.Value = Math.Max(numericUpDown_chunkSize.Minimum, _lastChunkSize / 2);
			}

			// Set last chunk size
			_lastChunkSize = (int) numericUpDown_chunkSize.Value;
		}

		private void button_fourier_Click(object sender, EventArgs e)
		{
			FourierTransform();
		}

		private void button_normalize_Click(object sender, EventArgs e)
		{
			// Abort if no track
			if (AH.CurrentTrack == null)
			{
				return;
			}

			// Normalize track via kernel or on host
			if (AH.CurrentTrack.OnHost)
			{
				// Normalize on host
				AH.CurrentTrack.Normalize();
			}
			else if (AH.CurrentTrack.OnDevice)
			{
				// Normalize via kernel
				// CCH.KernelH?.Normalize(AH.CurrentTrack.Pointer, AH.CurrentTrack.Length);
			}

			// Toggle UI
			ToggleUI();
		}

		private void button_compileAll_Click(object sender, EventArgs e)
		{
			// Call kernelH
			CCH.CompileAllKernels();

			// Toggle UI
			ToggleUI();
		}

		private void listBox_kernels_SelectedIndexChanged(object sender, EventArgs e)
		{
			// Abort if no KernelH
			if (CCH.KernelH == null)
			{
				return;
			}

			// Load selected entry by name
			string? name = listBox_kernels.SelectedItem?.ToString();
			if (!string.IsNullOrEmpty(name))
			{
				CCH.KernelH.LoadKernelByName(name);
			}

			// GuiB: BuildParams
			var parameters = CCH.KernelH.GetKernelParameters();
			GuiB.BuildParams(parameters, 5, 23, 0.45f);

			// Set kernel string
			if (string.IsNullOrEmpty(textBox_kernelString.Text) || CCH.KernelH?.KernelString != textBox_kernelString.Text)
			{
				// Set kernel string
				textBox_kernelString.Text = CCH.KernelH?.KernelString ?? "";
			}

			// Toggle UI
			ToggleUI();
		}

		private void button_kernelExecute_Click(object sender, EventArgs e)
		{
			// Abort if no kernel or track
			if (CCH.KernelH == null || CCH.KernelH.Kernel == null || AH.CurrentTrack == null || AH.CurrentTrack.Pointer == 0)
			{
				MessageBox.Show("No kernel or track loaded", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// Get kernel parameter values
			var pointer = AH.CurrentTrack.Pointer;
			var length = AH.CurrentTrack.Length;
			var parameters = GuiB.GetParamValues();

			// DEBUG log parameters with GuiB
			var logEntries = parameters.Select(p => $"{p} ({p.GetType().Name})");
			string logMessage = "Parameter values: " + string.Join(", ", logEntries);

			GuiB.Log(logMessage);

		}
	}
}
