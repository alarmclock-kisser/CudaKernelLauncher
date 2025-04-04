using NAudio.Wave;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Timer = System.Windows.Forms.Timer;

namespace CudaKernelLauncher
{
	public class AudioHandling
	{
		// ----- ATTRIBUTES ----- \\
		private ListBox TracksList;
		private PictureBox WaveBox;
		private Button PlaybackButton;
		private HScrollBar OffsetScrollbar;

		public List<AudioObject> Tracks = [];
		public int SamplesPerPixel { get; private set; } = 1024;
		public long Offset { get; private set; } = 0;

		private bool IsPlaying = false;

		// ----- LAMBDA ----- \\
		public AudioObject? CurrentTrack => this.TracksList.SelectedIndex >= 0 && this.TracksList.SelectedIndex < this.Tracks.Count ? this.Tracks[this.TracksList.SelectedIndex] : null;
		public Image? CurrentWave => this.DrawWaveform();

		public AudioObject? this[int index] => index >= 0 && index < this.Tracks.Count ? this.Tracks[index] : null;
		public AudioObject? this[string name] => this.Tracks.FirstOrDefault(t => t.Name == name);


		// ----- CONSTRUCTOR ----- \\
		public AudioHandling(ListBox? tracksList = null, PictureBox? view_pBox = null, Button? playback_button = null, HScrollBar? offset_scrollbar = null)
		{
			// Set attributes
			this.TracksList = tracksList ?? new ListBox();
			this.WaveBox = view_pBox ?? new PictureBox();
			this.PlaybackButton = playback_button ?? new Button();
			this.OffsetScrollbar = offset_scrollbar ?? new HScrollBar();

			// Register events
			this.WaveBox.MouseWheel += this.OnMouseWheel;
			this.TracksList.SelectedIndexChanged += (s, e) => this.ChangeTrack();
			this.PlaybackButton.Click += (s, e) => this.TogglePlayback();
			this.OffsetScrollbar.Scroll += (s, e) => this.SetOffset(this.OffsetScrollbar.Value);
			this.TracksList.MouseDown += (s, e) => this.RemoveTrack(this.TracksList.SelectedIndex);
		}


		// ----- METHODS ----- \\
		private void OnMouseWheel(object? sender, MouseEventArgs e)
		{
			if (this.CurrentTrack == null) return;

			if (Control.ModifierKeys == Keys.Control)
			{
				// Scroll Offset
				this.SetOffset(this.Offset - e.Delta * this.SamplesPerPixel / 8);
			}
			else
			{
				// Zoom mit Faktor 1.5 pro Stufe
				float zoomFactor = e.Delta < 0 ? 1.5f : 1 / 1.5f;
				this.SetZoom((int) (this.SamplesPerPixel * zoomFactor));
			}
		}

		private void SetZoom(int newZoom)
		{
			if (this.CurrentTrack == null) return;

			this.SamplesPerPixel = Math.Clamp(newZoom, 1, 16384);
			this.UpdateScrollbar();
			this.WaveBox.Image = this.CurrentWave;
		}

		private void SetOffset(long newOffset)
		{
			if (this.CurrentTrack == null) return;

			long maxOffset = Math.Max(0, this.CurrentTrack.Length - (this.WaveBox.Width * this.SamplesPerPixel));
			this.Offset = Math.Clamp(newOffset, 0, maxOffset);

			this.OffsetScrollbar.Value = (int) this.Offset;
			this.WaveBox.Image = this.CurrentWave;
		}

		private void ChangeTrack()
		{
			this.StopPlayback();
			this.Offset = 0;
			this.UpdateScrollbar();
			this.WaveBox.Image = this.CurrentWave;
		}

		private void UpdateScrollbar()
		{
			if (this.CurrentTrack == null) return;

			long maxOffset = Math.Max(0, this.CurrentTrack.Length - (this.WaveBox.Width * this.SamplesPerPixel));

			this.OffsetScrollbar.Minimum = 0;
			this.OffsetScrollbar.Maximum = (int) maxOffset;
			this.OffsetScrollbar.Value = (int) this.Offset;
			this.OffsetScrollbar.LargeChange = this.SamplesPerPixel;
		}

		private void TogglePlayback()
		{
			if (this.IsPlaying)
			{
				this.StopPlayback();
			}
			else
			{
				this.StartPlayback();
			}
		}

		private void StartPlayback()
		{
			if (this.CurrentTrack == null) return;

			this.CurrentTrack.PlayStop();
			this.IsPlaying = true;
			this.PlaybackButton.Text = "Stop";
		}

		private void StopPlayback()
		{
			if (this.CurrentTrack == null) return;

			this.CurrentTrack.Stop();
			this.IsPlaying = false;
			this.PlaybackButton.Text = "Play";
		}

		public void AddTrack(string path)
		{
			AudioObject track = new(path);
			this.Tracks.Add(track);
			this.FillTracksList();
		}

		public void RemoveTrack(int index)
		{
			// Only if NOT LEFT MOUSE BUTTON AND RIGHT MOUSE BUTTON
			if (!(Control.MouseButtons == MouseButtons.Right && Control.MouseButtons != MouseButtons.Left))
			{
				return;
			}

			if (index < 0 || index >= this.Tracks.Count)
			{
				return;
			}

			this.Tracks[index].Dispose();
			this.Tracks.RemoveAt(index);
			this.FillTracksList();
		}

		public void FillTracksList()
		{
			this.TracksList.Items.Clear();

			foreach (AudioObject track in this.Tracks)
			{
				string entry = (track.OnDevice ? "*" : "-") + " ";
				entry += track.Name.Length > 35 ? track.Name.Substring(0, 32) + "..." : track.Name;
				this.TracksList.Items.Add(entry);
			}
		}

		public Image? DrawWaveform()
		{
			// Get samplesperpixel
			int samplesPerPixel = this.CurrentTrack?.GetFitResolution(this.WaveBox.Width) ?? 1;
			return this.CurrentTrack?.GetWaveform(this.WaveBox, this.Offset, samplesPerPixel, Color.BlueViolet, Color.White);
		}
	}






	public class AudioObject
	{
		// ----- ATTRIBUTES ----- \\
		public string Name = "";
		public string Pth = "";

		public float[] Data = [];
		public long Pointer = 0;
		public char Form = 'f';
		public int Overlap = 0;

		public int Samplerate = 44100;
		public int Bitdepth = 24;
		public int Channels = 2;


		public WaveOutEvent Player = new();

		public Timer TimerPlayback = new();

		// ----- LAMBDA ----- \\
		public long Length => this.Data.LongLength;
		public double Duration => (double) this.Length / this.Samplerate / this.Channels / this.Bitdepth / 8;
		public string Meta => this.GetMeta();


		public bool Playing => this.Player.PlaybackState == PlaybackState.Playing;
		public long Position => this.Player.GetPosition();
		public double PositionSeconds => (double) this.Position / this.Samplerate / this.Channels / this.Bitdepth / 8;


		public bool OnHost => this.Data.LongLength > 0 && this.Pointer == 0;
		public bool OnDevice => this.Data.LongLength == 0 && this.Pointer != 0;


		// ----- CONSTRUCTOR ----- \\
		public AudioObject(string pth)
		{
			// Try load audio using naudio
			try
			{
				// Set path & name
				this.Pth = pth;
				this.Name = Path.GetFileName(pth);

				// Load audio
				using (AudioFileReader reader = new(pth))
				{
					// Set attributes
					this.Samplerate = reader.WaveFormat.SampleRate;
					this.Bitdepth = reader.WaveFormat.BitsPerSample;
					this.Channels = reader.WaveFormat.Channels;

					// Read data
					this.Data = new float[reader.Length];
					reader.Read(this.Data, 0, this.Data.Length);
				}
			}
			catch (Exception ex)
			{
				return;
			}
		}



		// ----- METHODS ----- \\
		// Dispose
		public void Dispose()
		{
			// Dispose data & reset pointer
			this.Data = [];
			this.Pointer = 0;
		}


		// Get Meta
		public string GetMeta()
		{
			StringBuilder sb = new();
			sb.Append(this.Samplerate / 1000 + " kHz, ");
			sb.Append(this.Bitdepth + " b, ");
			sb.Append(this.Channels + " ch, ");
			sb.Append((this.Data.Length / this.Samplerate / this.Channels / this.Bitdepth / 8) + " s, ");
			sb.Append(this.Data.LongLength + " samples");
			return sb.ToString();
		}


		// I/O
		public byte[] GetBytes()
		{
			int bytesPerSample = this.Bitdepth / 8;
			byte[] bytes = new byte[this.Data.Length * bytesPerSample];

			for (int i = 0; i < this.Data.Length; i++)
			{
				byte[] byteArray;
				float sample = this.Data[i];

				switch (this.Bitdepth)
				{
					case 16:
						short shortSample = (short) (sample * short.MaxValue);
						byteArray = BitConverter.GetBytes(shortSample);
						break;
					case 24:
						int intSample24 = (int) (sample * (1 << 23));
						byteArray = new byte[3];
						byteArray[0] = (byte) (intSample24 & 0xFF);
						byteArray[1] = (byte) ((intSample24 >> 8) & 0xFF);
						byteArray[2] = (byte) ((intSample24 >> 16) & 0xFF);
						break;
					case 32:
						int intSample32 = (int) (sample * int.MaxValue);
						byteArray = BitConverter.GetBytes(intSample32);
						break;
					default:
						throw new ArgumentException("Unsupported bit depth");
				}

				Buffer.BlockCopy(byteArray, 0, bytes, i * bytesPerSample, bytesPerSample);
			}

			return bytes;
		}

		public void ExportAudioWav(string filepath)
		{
			int sampleRate = this.Samplerate;
			int bitDepth = this.Bitdepth;
			int channels = this.Channels;
			float[] audioData = this.Data;

			// Berechne die tatsächliche Länge der Audiodaten
			int actualLength = audioData.Length / (bitDepth / 8) / channels;

			using (var fileStream = new FileStream(filepath, FileMode.Create))
			using (var writer = new BinaryWriter(fileStream))
			{
				// RIFF header
				writer.Write(Encoding.ASCII.GetBytes("RIFF"));
				writer.Write(36 + actualLength * channels * (bitDepth / 8)); // File size
				writer.Write(Encoding.ASCII.GetBytes("WAVE"));

				// fmt subchunk
				writer.Write(Encoding.ASCII.GetBytes("fmt "));
				writer.Write(16); // Subchunk1Size (16 for PCM)
				writer.Write((short) 1); // AudioFormat (1 for PCM)
				writer.Write((short) channels); // NumChannels
				writer.Write(sampleRate); // SampleRate
				writer.Write(sampleRate * channels * (bitDepth / 8)); // ByteRate
				writer.Write((short) (channels * (bitDepth / 8))); // BlockAlign
				writer.Write((short) bitDepth); // BitsPerSample

				// data subchunk
				writer.Write(Encoding.ASCII.GetBytes("data"));
				writer.Write(actualLength * channels * (bitDepth / 8)); // Subchunk2Size

				// Convert float array to the appropriate bit depth and write to file
				for (int i = 0; i < actualLength * channels; i++)
				{
					float sample = audioData[i];
					switch (bitDepth)
					{
						case 16:
							var shortSample = (short) (sample * short.MaxValue);
							writer.Write(shortSample);
							break;
						case 24:
							var intSample24 = (int) (sample * (1 << 23));
							writer.Write((byte) (intSample24 & 0xFF));
							writer.Write((byte) ((intSample24 >> 8) & 0xFF));
							writer.Write((byte) ((intSample24 >> 16) & 0xFF));
							break;
						case 32:
							var intSample32 = (int) (sample * int.MaxValue);
							writer.Write(intSample32);
							break;
						default:
							throw new ArgumentException("Unsupported bit depth");
					}
				}
			}
		}


		// Playback
		public void PlayStop(Button? playbackButton = null)
		{
			if (this.Player.PlaybackState == PlaybackState.Playing)
			{
				if (playbackButton != null)
				{
					playbackButton.Text = "⏵";
				}
				this.TimerPlayback.Stop();
				this.Player.Stop();
			}
			else
			{
				byte[] bytes = this.GetBytes();

				MemoryStream ms = new(bytes);
				RawSourceWaveStream raw = new(ms, new WaveFormat(this.Samplerate, this.Bitdepth, this.Channels));

				this.Player.Init(raw);

				if (playbackButton != null)
				{
					playbackButton.Text = "⏹";
				}
				this.TimerPlayback.Start();
				this.Player.Play();

				while (this.Player.PlaybackState == PlaybackState.Playing)
				{
					Application.DoEvents();
					Thread.Sleep(100);
				}

				if (playbackButton != null)
				{
					playbackButton.Text = "⏵";
				}
			}
		}

		public void Stop(Button? playbackButton = null)
		{
			if (this.Player.PlaybackState == PlaybackState.Playing)
			{
				if (playbackButton != null)
				{
					playbackButton.Text = "⏵";
				}
				this.TimerPlayback.Stop();
				this.Player.Stop();
			}
		}


		// Normalize
		public void Normalize(float target = 1)
		{
			// Abort if no data or playing
			if (this.Data.Length == 0 || this.Playing)
			{
				return;
			}

			// Get max value
			float max = this.Data.Max(Math.Abs);

			// Normalize (Parallel)
			Parallel.For(0, this.Data.Length, i =>
			{
				this.Data[i] = this.Data[i] / max * target;
			});
		}


		// Waveform
		public Bitmap GetWaveform(PictureBox waveBox, long offset = 0, int samplesPerPixel = 1, Color? graphColor = null, Color? bgColor = null)
		{
			// Determine offset
			offset = Math.Clamp(offset, 0, this.Data.LongLength);

			// Validate inputs
			if (this.Data.Length == 0 || waveBox.Width <= 0 || waveBox.Height <= 0)
			{
				return new Bitmap(1, 1);
			}
			// Set colors
			Color waveColor = graphColor ?? Color.Blue;
			Color bg = bgColor ?? (waveColor.GetBrightness() > 0.5f ? Color.White : Color.Black);

			// Create bitmap & graphics
			Bitmap bmp = new(waveBox.Width, waveBox.Height);
			using Graphics g = Graphics.FromImage(bmp);
			g.SmoothingMode = SmoothingMode.AntiAlias;
			g.Clear(bg);
			using Pen pen = new(waveColor);

			// Y-axis settings
			float centerY = waveBox.Height / 2f;
			float scale = centerY;

			// Draw waveform
			for (int x = 0; x < waveBox.Width; x++)
			{
				long sampleIdx = offset + x * samplesPerPixel;
				if (sampleIdx >= this.Data.Length) break;

				float min = float.MaxValue, max = float.MinValue;

				for (int i = 0; i < samplesPerPixel && sampleIdx + i < this.Data.Length; i++)
				{
					float sample = this.Data[sampleIdx + i];
					if (sample > max) max = sample;
					if (sample < min) min = sample;
				}

				float yMax = centerY - max * scale;
				float yMin = centerY - min * scale;

				g.DrawLine(pen, x, Math.Clamp(yMax, 0, waveBox.Height), x, Math.Clamp(yMin, 0, waveBox.Height));
			}

			return bmp;
		}

		public int GetFitResolution(int width)
		{
			// Use the number of channels to ensure correct scaling
			int totalSamples = this.Data.Length / this.Channels / (this.Bitdepth / 8);
			return Math.Max(1, (int) Math.Ceiling((double) totalSamples / width));
		}


		// Chunking
		public List<float[]> MakeChunks(int chunkSize, int overlap = 0)
		{
			// If overlap is 0 take half of chunk size
			overlap = overlap == 0 ? chunkSize / 2 : overlap;

			// Verify overlap
			this.Overlap = Math.Max(0, Math.Min(overlap, chunkSize / 2));

			// Get chunk count & make chunks List
			int stepSize = chunkSize - this.Overlap;
			int chunkCount = (int) Math.Ceiling((double) this.Data.Length / stepSize);

			List<float[]> chunks = new(chunkCount);
			int index = 0;

			while (index < this.Data.Length)
			{
				// Set default chunk size
				int length = Math.Min(chunkSize, this.Data.Length - index);
				float[] chunk = new float[chunkSize]; // Immer volle Größe

				// Copy data to chunk
				Array.Copy(this.Data, index, chunk, 0, length);
				chunks.Add(chunk);

				// Increase index
				index += stepSize;
			}

			return chunks;
		}

		public void AggregateChunks(List<float[]> chunks)
		{
			if (chunks == null || chunks.Count == 0)
			{
				this.Data = [];
				return;
			}

			// Get step size & total length
			int stepSize = chunks[0].Length - this.Overlap;
			int totalLength = stepSize * (chunks.Count - 1) + chunks[^1].Length;
			float[] aggregated = new float[totalLength];

			int index = 0;
			foreach (float[] chunk in chunks)
			{
				// Get copy length (min of chunk length or remaining space)
				int copyLength = Math.Min(chunk.Length, aggregated.Length - index);

				Array.Copy(chunk, 0, aggregated, index, copyLength);
				index += stepSize;
			}

			// Set Floats & Length
			this.Data = aggregated;
		}


		// Debug
		public string GetFirstSamples(int count = 0, bool skipSilence = false)
		{
			// Set count if not set
			count = count == 0 ? this.Samplerate : count;

			// Abort if no data
			if (this.Data.Length == 0)
			{
				return "";
			}

			// Skip silence
			int start = 0;
			var data = this.Data;
			if (skipSilence)
			{
				while (start < this.Data.Length && Math.Abs(this.Data[start]) < 0.01f)
				{
					start++;
				}
				count = Math.Min(count, this.Data.Length - start);
				data = this.Data.Skip(start).Take(count).ToArray();
			}

			// Get samples
			StringBuilder sb = new();
			sb.Append("[");
			for (int i = 0; i < count; i++)
			{
				sb.Append(data[i] + ", ");
			}
			sb.Remove(sb.Length - 2, 2);
			sb.Append("]");
			return sb.ToString();
		}


	}
}
