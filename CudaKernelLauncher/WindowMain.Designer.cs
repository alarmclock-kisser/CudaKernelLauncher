namespace CudaKernelLauncher
{
    partial class WindowMain
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

		#region Windows Form Designer generated code

		/// <summary>
		///  Required method for Designer support - do not modify
		///  the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.comboBox_devices = new ComboBox();
			this.listBox_log = new ListBox();
			this.pictureBox_wave = new PictureBox();
			this.listBox_tracks = new ListBox();
			this.hScrollBar_offset = new HScrollBar();
			this.button_play = new Button();
			this.button_export = new Button();
			this.button_import = new Button();
			this.label_vram = new Label();
			this.progressBar_vram = new ProgressBar();
			this.listBox_kernels = new ListBox();
			this.button_move = new Button();
			this.groupBox_controls = new GroupBox();
			this.panel_toggleChunking = new Panel();
			this.numericUpDown_chunkSize = new NumericUpDown();
			this.groupBox_transform = new GroupBox();
			this.button_normalize = new Button();
			this.button_fourier = new Button();
			this.groupBox_kernels = new GroupBox();
			this.button_compileAll = new Button();
			this.label_kernelLoaded = new Label();
			this.panel_kernelParams = new Panel();
			this.textBox_kernelString = new TextBox();
			this.button_kernelCompile = new Button();
			this.label_kernelInfo = new Label();
			this.button_kernelExecute = new Button();
			((System.ComponentModel.ISupportInitialize) this.pictureBox_wave).BeginInit();
			this.groupBox_controls.SuspendLayout();
			this.panel_toggleChunking.SuspendLayout();
			((System.ComponentModel.ISupportInitialize) this.numericUpDown_chunkSize).BeginInit();
			this.groupBox_transform.SuspendLayout();
			this.groupBox_kernels.SuspendLayout();
			this.SuspendLayout();
			// 
			// comboBox_devices
			// 
			this.comboBox_devices.FormattingEnabled = true;
			this.comboBox_devices.Location = new Point(12, 12);
			this.comboBox_devices.Name = "comboBox_devices";
			this.comboBox_devices.Size = new Size(260, 23);
			this.comboBox_devices.TabIndex = 0;
			// 
			// listBox_log
			// 
			this.listBox_log.FormattingEnabled = true;
			this.listBox_log.ItemHeight = 15;
			this.listBox_log.Location = new Point(12, 796);
			this.listBox_log.Name = "listBox_log";
			this.listBox_log.Size = new Size(776, 169);
			this.listBox_log.TabIndex = 1;
			// 
			// pictureBox_wave
			// 
			this.pictureBox_wave.Location = new Point(12, 621);
			this.pictureBox_wave.Name = "pictureBox_wave";
			this.pictureBox_wave.Size = new Size(610, 152);
			this.pictureBox_wave.TabIndex = 2;
			this.pictureBox_wave.TabStop = false;
			// 
			// listBox_tracks
			// 
			this.listBox_tracks.FormattingEnabled = true;
			this.listBox_tracks.ItemHeight = 15;
			this.listBox_tracks.Location = new Point(628, 621);
			this.listBox_tracks.Name = "listBox_tracks";
			this.listBox_tracks.Size = new Size(160, 169);
			this.listBox_tracks.TabIndex = 3;
			this.listBox_tracks.SelectedIndexChanged += this.listBox_tracks_SelectedIndexChanged;
			// 
			// hScrollBar_offset
			// 
			this.hScrollBar_offset.Location = new Point(12, 776);
			this.hScrollBar_offset.Name = "hScrollBar_offset";
			this.hScrollBar_offset.Size = new Size(610, 17);
			this.hScrollBar_offset.TabIndex = 4;
			// 
			// button_play
			// 
			this.button_play.Location = new Point(6, 100);
			this.button_play.Name = "button_play";
			this.button_play.Size = new Size(23, 23);
			this.button_play.TabIndex = 5;
			this.button_play.Text = ">";
			this.button_play.UseVisualStyleBackColor = true;
			// 
			// button_export
			// 
			this.button_export.Location = new Point(79, 100);
			this.button_export.Name = "button_export";
			this.button_export.Size = new Size(75, 23);
			this.button_export.TabIndex = 6;
			this.button_export.Text = "Export";
			this.button_export.UseVisualStyleBackColor = true;
			this.button_export.Click += this.button_export_Click;
			// 
			// button_import
			// 
			this.button_import.Location = new Point(79, 71);
			this.button_import.Name = "button_import";
			this.button_import.Size = new Size(75, 23);
			this.button_import.TabIndex = 7;
			this.button_import.Text = "Import";
			this.button_import.UseVisualStyleBackColor = true;
			this.button_import.Click += this.button_import_Click;
			// 
			// label_vram
			// 
			this.label_vram.AutoSize = true;
			this.label_vram.Location = new Point(12, 38);
			this.label_vram.Name = "label_vram";
			this.label_vram.Size = new Size(90, 15);
			this.label_vram.TabIndex = 8;
			this.label_vram.Text = "VRAM: 0 / 0 MB";
			// 
			// progressBar_vram
			// 
			this.progressBar_vram.Location = new Point(12, 56);
			this.progressBar_vram.Name = "progressBar_vram";
			this.progressBar_vram.Size = new Size(260, 12);
			this.progressBar_vram.TabIndex = 9;
			// 
			// listBox_kernels
			// 
			this.listBox_kernels.FormattingEnabled = true;
			this.listBox_kernels.ItemHeight = 15;
			this.listBox_kernels.Location = new Point(628, 27);
			this.listBox_kernels.Name = "listBox_kernels";
			this.listBox_kernels.Size = new Size(160, 214);
			this.listBox_kernels.TabIndex = 10;
			this.listBox_kernels.SelectedIndexChanged += this.listBox_kernels_SelectedIndexChanged;
			// 
			// button_move
			// 
			this.button_move.Location = new Point(6, 22);
			this.button_move.Name = "button_move";
			this.button_move.Size = new Size(60, 23);
			this.button_move.TabIndex = 11;
			this.button_move.Text = "Move";
			this.button_move.UseVisualStyleBackColor = true;
			this.button_move.Click += this.button_move_Click;
			// 
			// groupBox_controls
			// 
			this.groupBox_controls.Controls.Add(this.button_move);
			this.groupBox_controls.Controls.Add(this.panel_toggleChunking);
			this.groupBox_controls.Controls.Add(this.button_import);
			this.groupBox_controls.Controls.Add(this.button_export);
			this.groupBox_controls.Controls.Add(this.button_play);
			this.groupBox_controls.Location = new Point(628, 486);
			this.groupBox_controls.Name = "groupBox_controls";
			this.groupBox_controls.Size = new Size(160, 129);
			this.groupBox_controls.TabIndex = 12;
			this.groupBox_controls.TabStop = false;
			this.groupBox_controls.Text = "Controls";
			// 
			// panel_toggleChunking
			// 
			this.panel_toggleChunking.Controls.Add(this.numericUpDown_chunkSize);
			this.panel_toggleChunking.Location = new Point(72, 22);
			this.panel_toggleChunking.Name = "panel_toggleChunking";
			this.panel_toggleChunking.Size = new Size(82, 23);
			this.panel_toggleChunking.TabIndex = 13;
			// 
			// numericUpDown_chunkSize
			// 
			this.numericUpDown_chunkSize.Location = new Point(0, 0);
			this.numericUpDown_chunkSize.Maximum = new decimal(new int[] { 4194304, 0, 0, 0 });
			this.numericUpDown_chunkSize.Minimum = new decimal(new int[] { 256, 0, 0, 0 });
			this.numericUpDown_chunkSize.Name = "numericUpDown_chunkSize";
			this.numericUpDown_chunkSize.Size = new Size(82, 23);
			this.numericUpDown_chunkSize.TabIndex = 12;
			this.numericUpDown_chunkSize.Value = new decimal(new int[] { 256, 0, 0, 0 });
			this.numericUpDown_chunkSize.ValueChanged += this.numericUpDown_chunkSize_ValueChanged;
			// 
			// groupBox_transform
			// 
			this.groupBox_transform.Controls.Add(this.button_normalize);
			this.groupBox_transform.Controls.Add(this.button_fourier);
			this.groupBox_transform.Location = new Point(628, 422);
			this.groupBox_transform.Name = "groupBox_transform";
			this.groupBox_transform.Size = new Size(160, 58);
			this.groupBox_transform.TabIndex = 13;
			this.groupBox_transform.TabStop = false;
			this.groupBox_transform.Text = "CUDA Transform";
			// 
			// button_normalize
			// 
			this.button_normalize.Location = new Point(72, 22);
			this.button_normalize.Name = "button_normalize";
			this.button_normalize.Size = new Size(82, 23);
			this.button_normalize.TabIndex = 14;
			this.button_normalize.Text = "Normalize";
			this.button_normalize.UseVisualStyleBackColor = true;
			this.button_normalize.Click += this.button_normalize_Click;
			// 
			// button_fourier
			// 
			this.button_fourier.Location = new Point(6, 22);
			this.button_fourier.Name = "button_fourier";
			this.button_fourier.Size = new Size(60, 23);
			this.button_fourier.TabIndex = 14;
			this.button_fourier.Text = "Fourier";
			this.button_fourier.UseVisualStyleBackColor = true;
			this.button_fourier.Click += this.button_fourier_Click;
			// 
			// groupBox_kernels
			// 
			this.groupBox_kernels.Controls.Add(this.button_kernelExecute);
			this.groupBox_kernels.Controls.Add(this.button_compileAll);
			this.groupBox_kernels.Location = new Point(628, 267);
			this.groupBox_kernels.Name = "groupBox_kernels";
			this.groupBox_kernels.Size = new Size(160, 149);
			this.groupBox_kernels.TabIndex = 14;
			this.groupBox_kernels.TabStop = false;
			this.groupBox_kernels.Text = "CUDA Kernels";
			// 
			// button_compileAll
			// 
			this.button_compileAll.Location = new Point(6, 120);
			this.button_compileAll.Name = "button_compileAll";
			this.button_compileAll.Size = new Size(75, 23);
			this.button_compileAll.TabIndex = 15;
			this.button_compileAll.Text = "Compile all";
			this.button_compileAll.UseVisualStyleBackColor = true;
			this.button_compileAll.Click += this.button_compileAll_Click;
			// 
			// label_kernelLoaded
			// 
			this.label_kernelLoaded.AutoSize = true;
			this.label_kernelLoaded.Location = new Point(628, 9);
			this.label_kernelLoaded.Name = "label_kernelLoaded";
			this.label_kernelLoaded.Size = new Size(81, 15);
			this.label_kernelLoaded.TabIndex = 15;
			this.label_kernelLoaded.Text = "Loaded kernel";
			// 
			// panel_kernelParams
			// 
			this.panel_kernelParams.BackColor = Color.White;
			this.panel_kernelParams.Location = new Point(278, 9);
			this.panel_kernelParams.Name = "panel_kernelParams";
			this.panel_kernelParams.Size = new Size(344, 232);
			this.panel_kernelParams.TabIndex = 16;
			// 
			// textBox_kernelString
			// 
			this.textBox_kernelString.AcceptsReturn = true;
			this.textBox_kernelString.AcceptsTab = true;
			this.textBox_kernelString.Font = new Font("Bahnschrift SemiCondensed", 8.25F, FontStyle.Regular, GraphicsUnit.Point,  0);
			this.textBox_kernelString.Location = new Point(12, 267);
			this.textBox_kernelString.MaxLength = 9999999;
			this.textBox_kernelString.Multiline = true;
			this.textBox_kernelString.Name = "textBox_kernelString";
			this.textBox_kernelString.PlaceholderText = "Kernel string here";
			this.textBox_kernelString.Size = new Size(610, 313);
			this.textBox_kernelString.TabIndex = 17;
			// 
			// button_kernelCompile
			// 
			this.button_kernelCompile.Location = new Point(547, 586);
			this.button_kernelCompile.Name = "button_kernelCompile";
			this.button_kernelCompile.Size = new Size(75, 23);
			this.button_kernelCompile.TabIndex = 18;
			this.button_kernelCompile.Text = "Compile";
			this.button_kernelCompile.UseVisualStyleBackColor = true;
			// 
			// label_kernelInfo
			// 
			this.label_kernelInfo.AutoSize = true;
			this.label_kernelInfo.Location = new Point(12, 583);
			this.label_kernelInfo.Name = "label_kernelInfo";
			this.label_kernelInfo.Size = new Size(64, 15);
			this.label_kernelInfo.TabIndex = 19;
			this.label_kernelInfo.Text = "Kernel info";
			// 
			// button_kernelExecute
			// 
			this.button_kernelExecute.Location = new Point(87, 120);
			this.button_kernelExecute.Name = "button_kernelExecute";
			this.button_kernelExecute.Size = new Size(67, 23);
			this.button_kernelExecute.TabIndex = 16;
			this.button_kernelExecute.Text = "Execute";
			this.button_kernelExecute.UseVisualStyleBackColor = true;
			this.button_kernelExecute.Click += this.button_kernelExecute_Click;
			// 
			// WindowMain
			// 
			this.AutoScaleDimensions = new SizeF(7F, 15F);
			this.AutoScaleMode = AutoScaleMode.Font;
			this.ClientSize = new Size(800, 977);
			this.Controls.Add(this.label_kernelInfo);
			this.Controls.Add(this.button_kernelCompile);
			this.Controls.Add(this.textBox_kernelString);
			this.Controls.Add(this.panel_kernelParams);
			this.Controls.Add(this.label_kernelLoaded);
			this.Controls.Add(this.groupBox_kernels);
			this.Controls.Add(this.groupBox_transform);
			this.Controls.Add(this.groupBox_controls);
			this.Controls.Add(this.listBox_kernels);
			this.Controls.Add(this.progressBar_vram);
			this.Controls.Add(this.label_vram);
			this.Controls.Add(this.hScrollBar_offset);
			this.Controls.Add(this.listBox_tracks);
			this.Controls.Add(this.pictureBox_wave);
			this.Controls.Add(this.listBox_log);
			this.Controls.Add(this.comboBox_devices);
			this.Name = "WindowMain";
			this.Text = "CUDA Kernel Launcher for Audio Processing";
			((System.ComponentModel.ISupportInitialize) this.pictureBox_wave).EndInit();
			this.groupBox_controls.ResumeLayout(false);
			this.panel_toggleChunking.ResumeLayout(false);
			((System.ComponentModel.ISupportInitialize) this.numericUpDown_chunkSize).EndInit();
			this.groupBox_transform.ResumeLayout(false);
			this.groupBox_kernels.ResumeLayout(false);
			this.ResumeLayout(false);
			this.PerformLayout();
		}

		#endregion

		private ComboBox comboBox_devices;
		private ListBox listBox_log;
		private PictureBox pictureBox_wave;
		private ListBox listBox_tracks;
		private HScrollBar hScrollBar_offset;
		private Button button_play;
		private Button button_export;
		private Button button_import;
		private Label label_vram;
		private ProgressBar progressBar_vram;
		private ListBox listBox_kernels;
		private Button button_move;
		private GroupBox groupBox_controls;
		private NumericUpDown numericUpDown_chunkSize;
		private Panel panel_toggleChunking;
		private GroupBox groupBox_transform;
		private Button button_fourier;
		private Button button_normalize;
		private GroupBox groupBox_kernels;
		private Button button_compileAll;
		private Label label_kernelLoaded;
		private Panel panel_kernelParams;
		private TextBox textBox_kernelString;
		private Button button_kernelCompile;
		private Label label_kernelInfo;
		private Button button_kernelExecute;
	}
}
