using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaFFT;
using ManagedCuda.VectorTypes;

namespace CudaKernelLauncher
{
	public class CudaFftHandling
	{
		// ----- ATTRIBUTES ----- \\
		public CudaContextHandling CudaH;
		private PrimaryContext Ctx;


		// ----- OBJECTS ----- \\





		// ----- LAMBDA ----- \\
		public CudaMemoryHandling? MemH => CudaH.MemH;
		public List<CUdeviceptr[]> Buffers => MemH?.Buffers.Keys.ToList() ?? [];
		public List<int[]> BufferSizes => MemH?.Buffers.Values.ToList() ?? [];
		public long[] IndexPointers => MemH?.IndexPointers ?? [];


		public int LogInterval => CudaH.LogInterval;


		// ----- CONSTRUCTOR ----- \\
		public CudaFftHandling(CudaContextHandling cudaH, PrimaryContext ctx)
		{
			// Set attributes
			this.CudaH = cudaH;
			this.Ctx = ctx;


			// Register events


		}




		// ----- METHODS ----- \\
		// Log
		public void Log(string message, string inner = "", int level = 1, bool update = false)
		{
			string msg = "[" + DateTime.Now.ToString("hh:mm:ss.fff") + "] ";
			msg += "<FFT> ";
			for (int i = 0; i < level; i++)
			{
				msg += " - ";
			}
			msg += message;
			if (!string.IsNullOrEmpty(inner))
			{
				msg += " (" + inner + ")";
			}

			if (update)
			{
				CudaH.LogBox.Items[CudaH.LogBox.Items.Count - 1] = msg;
			}
			else
			{
				CudaH.LogBox.Items.Add(msg);
			}


		}


		// Dispose
		public void Dispose()
		{
			// Dispose ...
		}


		// FFT
		public long PerformFFT(long indexPointer, bool keep = false, bool silent = false)
		{
			// Abort if no memory handling
			if (MemH == null)
			{
				if (!silent)
				{
					Log("No memory handling detected", "", 1);
				}
				return 0;
			}

			// Get buffers & sizes
			CUdeviceptr[] buffers = MemH?[indexPointer] ?? [];
			int[] sizes = MemH?.GetSizesFromIndex(indexPointer) ?? [];

			// Abort if any int size is < 1 or if buffers are null
			if (buffers.LongLength == 0 || sizes.LongLength == 0 || sizes.Any(x => x < 1))
			{
				if (!silent)
				{
					Log("Invalid buffers / sizes detected", "Count: " + buffers?.Length, 1);
				}
				return indexPointer;
			}

			// Make buffers list for results (float2)
			CUdeviceptr[] results = new CUdeviceptr[buffers.LongLength];

			// Pre log
			if (!silent)
			{
				this.Log("");
			}

			// Perform FFT (forwards) for each buffer with the corresponding size
			for (int i = 0; i < buffers.LongLength; i++)
			{
				// Allocate result buffer
				results[i] = new CudaDeviceVariable<float2>(sizes[i]).DevicePointer;

				// Create plan
				CudaFFTPlan1D plan = new(sizes[i], cufftType.R2C, 1);
				
				// Execute plan
				plan.Exec(buffers[i], results[i]);
				
				// Dispose plan
				plan.Dispose();

				// Log progress every log interval
				if (i % LogInterval == 0 && !silent)
				{
					Log("FFT performed", "Buffer: " + i + " / " + buffers.LongLength, 2, true);
				}
			}

			// Optionally keep buffers
			if (!keep)
			{
				MemH?.FreePointerGroup(indexPointer, true);
			}

			// Get index pointer for results
			long indexPointerResults = results.FirstOrDefault().Pointer;

			// Add results to MemH
			MemH?.Buffers.Add(results, sizes);

			// Log success
			if (!silent)
			{
				Log("FFT completed", "Buffers: " + buffers.LongLength + ", Ptr: " + indexPointer, 1, true);
			}

			// Return index pointer
			return indexPointerResults;
		}

		public long PerformIFFT(long indexPointer, bool keep = false, bool silent = false)
		{
			// Abort if no memory handling
			if (MemH == null)
			{
				if (!silent)
				{
					Log("No memory handling detected", "", 1);
				}
				return 0;
			}

			// Get buffers & sizes
			CUdeviceptr[] buffers = MemH?[indexPointer] ?? [];
			int[] sizes = MemH?.GetSizesFromIndex(indexPointer) ?? [];

			// Abort if any int size is < 1 or if buffers are null
			if (buffers.LongLength == 0 || sizes.LongLength == 0 || sizes.Any(x => x < 1))
			{
				if (!silent)
				{
					Log("Invalid buffers / sizes detected", "Count: " + buffers?.Length, 1);
				}
				return indexPointer;
			}

			// Make buffers list for results (float2)
			CUdeviceptr[] results = new CUdeviceptr[buffers.LongLength];

			// Pre log
			if (!silent)
			{
				this.Log("");
			}

			// Perform FFT (forwards) for each buffer with the corresponding size
			for (int i = 0; i < buffers.LongLength; i++)
			{
				// Allocate result buffer
				results[i] = new CudaDeviceVariable<float>(sizes[i]).DevicePointer;

				// Create plan
				CudaFFTPlan1D plan = new(sizes[i], cufftType.C2R, 1);

				// Execute plan
				plan.Exec(buffers[i], results[i]);

				// Dispose plan
				plan.Dispose();

				// Log progress every log interval
				if (i % LogInterval == 0 && !silent)
				{
					Log("I-FFT performed", "Buffer: " + i + " / " + buffers.LongLength, 2, true);
				}
			}

			// Optionally keep buffers
			if (!keep)
			{
				MemH?.FreePointerGroup(indexPointer, true);
			}

			// Get index pointer for results
			long indexPointerResults = results.FirstOrDefault().Pointer;

			// Add results to MemH
			MemH?.Buffers.Add(results, sizes);

			// Log success
			if (!silent)
			{
				Log("I-FFT completed", "Buffers: " + buffers.LongLength + ", Ptr: " + indexPointer, 1, true);
			}

			// Return index pointer
			return indexPointerResults;
		}


		// FFT (many)
		public long[] PerformFFTMany(long[] indexPointers, bool keep = false, bool silent = false)
		{
			// Abort if no memory handling
			if (MemH == null)
			{
				if (!silent)
				{
					Log("No memory handling detected", "", 1);
				}
				return [];
			}

			// Get buffers & sizes
			CUdeviceptr[]?[] buffers = indexPointers.Select(x => MemH?[x]).ToArray() ?? [];
			int[]?[] sizes = indexPointers.Select(x => MemH?.GetSizesFromIndex(x)).ToArray() ?? [];

			// Abort if any int size is < 1 or if buffers are null
			if (buffers.LongLength == 0 || sizes.LongLength == 0 || sizes.Any(x => x == null || x.LongLength == 0 || x.Any(y => y < 1)))
			{
				if (!silent)
				{
					Log("Invalid buffers / sizes detected", "Count: " + buffers?.Length, 1);
				}
				return indexPointers;
			}

			// Make buffers list for results (float2)
			CUdeviceptr[] results = new CUdeviceptr[buffers.LongLength];
			int index = 0;

			// Pre log
			if (!silent)
			{
				this.Log("");
			}

			// Perform FFT many (forwards) with each buffer with the corresponding size




			// Optionally keep buffers
			if (!keep)
			{
				foreach (long indexPointer in indexPointers)
				{
					MemH?.FreePointerGroup(indexPointer, true);
				}
			}

			// Get index pointers for results



			// Add results to MemH
			MemH?.Buffers.Add(results, sizes.Select(x => x?.FirstOrDefault() ?? 0).ToArray());

			// Log success
			if (!silent)
			{
				Log("FFT many completed", "Buffers: " + buffers.LongLength + ", Ptr: " + indexPointers, 1, true);
			}

			// Return index pointers
			return indexPointers;
		}
	}
}
