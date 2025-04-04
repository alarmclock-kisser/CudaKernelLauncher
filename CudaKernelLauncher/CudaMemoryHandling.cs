using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace CudaKernelLauncher
{
	public class CudaMemoryHandling
	{
		// ----- ATTRIBUTES ----- \\
		private CudaContextHandling CudaH;
		private PrimaryContext Ctx;

		public Dictionary<CUdeviceptr[], int[]> Buffers = [];




		// ----- OBJECTS ----- \\



		// ----- LAMBDA ----- \\
		public long[] IndexPointers => GetIndexPointers();

		public CUdeviceptr[] this[long indexPointer]
		{
			get => this.GetPointersFromIndex(indexPointer, out _) ?? [];
		}

		public List<int[]> Sizes => Buffers.Values.ToList();

		public CUdeviceptr[] this[int index]
		{
			get => index > 0 && index < Buffers.Count ? Buffers.ElementAt(index).Key : [];
		}

		public int LogInterval => CudaH.LogInterval;

		// ----- CONSTRUCTOR ----- \\
		public CudaMemoryHandling(CudaContextHandling cudaHandling, PrimaryContext ctx)
		{
			// Set attributes
			this.CudaH = cudaHandling;
			this.Ctx = ctx;


		}






		// ----- METHODS ----- \\
		public void Log(string message, string inner = "", int level = 1, bool update = false)
		{
			string msg = "[" + DateTime.Now.ToString("hh:mm:ss.fff") + "] ";
			msg += "<Mem> ";
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
				// Overwrite last Output Line with new message
				CudaH.LogBox.Items[^1] = msg;
			}
			else
			{
				CudaH.LogBox.Items.Add(msg);
			}


		}

		public void Dispose()
		{
			// Free every buffer (group)
			foreach (long indexPointer in IndexPointers)
			{
				FreePointerGroup(indexPointer, true);
			}

			// Clear buffers registry
			Buffers.Clear();

		}

		public long[] GetIndexPointers()
		{
			long[] ptrs = new long[Buffers.Count];

			for (int i = 0; i < Buffers.Count; i++)
			{
				// Get first pointer from every buffer group
				var ptr = Buffers.ElementAt(i).Key.FirstOrDefault();

				// Add to array
				ptrs[i] = ptr.Pointer;
			}

			return ptrs;
		}

		public CUdeviceptr[] GetPointersFromIndex(long indexPointer, out int[] sizes)
		{
			// Lookup index pointer
			if (!IndexPointers.Contains(indexPointer))
			{
				sizes = [];
				CudaH.Log("Index pointer not found", "Ptr: " + indexPointer, 2);
				return [];
			}

			// Try get buffer group by where index pointer is FirstOrDefault.Pointer
			var bufferGroup = Buffers.FirstOrDefault(x => x.Key.FirstOrDefault().Pointer == indexPointer);
			sizes = bufferGroup.Value;
			CUdeviceptr[] pointers = bufferGroup.Key;

			return pointers;
		}

		public int[] GetSizesFromIndex(long indexPointer)
		{
			// Lookup index pointer
			if (!IndexPointers.Contains(indexPointer))
			{
				CudaH.Log("Index pointer not found", "Ptr: " + indexPointer, 2);
				return [];
			}
			
			// Try get buffer group by where index pointer is FirstOrDefault.Pointer
			var bufferGroup = Buffers.FirstOrDefault(x => x.Key.FirstOrDefault().Pointer == indexPointer);
			int[] sizes = bufferGroup.Value;
			return sizes;
		}

		public Type? GetTypeFromIndex(long indexPointer, bool silent = false)
		{
			// Lookup index pointer
			if (!IndexPointers.Contains(indexPointer))
			{
				if (!silent)
				{
					CudaH.Log("Index pointer not found", "Ptr: " + indexPointer, 2);
				}
				return null;
			}
			
			// Get pointers from index pointer
			var pointers = GetPointersFromIndex(indexPointer, out _);
			
			// Get type from first pointer
			Type? type = pointers.FirstOrDefault().AttributeMemoryType.GetType();
			return type;
		}

		public void FreePointerGroup(long indexPointer, bool silent = false)
		{
			// Get pointers from index pointer
			var pointers = GetPointersFromIndex(indexPointer, out int[] sizes);

			// Abort if no pointers
			if (pointers.Length == 0)
			{
				if (!silent)
				{
					CudaH.Log("No pointers found for index pointer", "Ptr: " + indexPointer, 2);
				}
				return;
			}

			// Free every pointer
			foreach (var ptr in pointers)
			{
				try
				{
					Ctx.FreeMemory(ptr);
				}
				catch (Exception e)
				{
					if (!silent)
					{
						CudaH.Log("Failed to free memory", e.Message, 2);
					}
				}
			}

			// Remove from Buffers
			Buffers.Remove(pointers);

			// Log
			if (!silent)
			{
				CudaH.Log("Pointer group freed", "Ptr: " + indexPointer, 1);
			}
		}

		public List<T[]> MakeChunks<T>(T[] data, int chunkSize = 0) where T : unmanaged
		{
			// Abort if no data
			if (data.LongLength == 0)
			{
				this.Log("No data to make chunks of", "Data.LongLength: " + data.LongLength, 1);
				return [];
			}

			// Adjust chunk size if 0 or higher than data length
			if (chunkSize == 0 || chunkSize > data.LongLength)
			{
				// If higher than int.MaxValue, set to int.MaxValue
				if (data.LongLength > int.MaxValue)
				{
					chunkSize = int.MaxValue;
				}
				else
				{
					chunkSize = (int) data.LongLength;
				}

				// Log
				this.Log("Chunk size adjusted", "ChunkSize: " + chunkSize, 2);
			}
			Math.Clamp(chunkSize, data.LongLength, int.MaxValue);

			// Get chunk count & make chunks List
			int chunkCount = (int) Math.Ceiling((double) data.LongLength / chunkSize);
			List<T[]> chunks = [];

			// Split data into chunks, fill last chunk with 0s
			for (int i = 0; i < chunkCount; i++)
			{
				// Create chunk with chunkSize length
				T[] chunk = new T[chunkSize];

				// Copy data to chunk
				chunk = data.Skip(i * chunkSize).Take(chunkSize).ToArray() ?? new T[chunkSize];

				// Resize chunk if last chunk & smaller than chunkSize
				if (i == chunkCount - 1 && chunk.Length < chunkSize)
				{
					Array.Resize(ref chunk, chunkSize);

					// Fill last with default value
					for (int j = data.Length; j < chunkSize; j++)
					{
						chunk[j] = default;
					}
				}

				// Add chunk to chunks
				chunks.Add(chunk);
			}

			// Log
			this.Log("Data split into chunks", "ChunkCount: " + chunkCount + " ,ChunkSize: " + chunkSize, 1);

			// Return List
			return chunks;
		}

		public long PushData<T>(T[] data, int chunkSize, bool silent = false) where T : unmanaged
		{
			var chunks = MakeChunks(data, chunkSize);

			// Abort if no chunks
			if (chunks.Count == 0 && !silent)
			{
				this.Log("No chunks to push data", "Chunks.Count: " + chunks.Count, 1);
				return 0;
			}

			// Create CudaDeviceVariable array & int[]
			CudaDeviceVariable<T>[] buffers = new CudaDeviceVariable<T>[chunks.Count];
			int[] sizes = new int[chunks.Count];
			CUdeviceptr[] pointers = new CUdeviceptr[chunks.Count];

			// Pre log
			if (!silent)
			{
				this.Log("Pushing data to device", "Chunks.Count: " + chunks.Count, 1);
				this.Log("");
			}

			// Allocate memory for every chunk
			for (int i = 0; i < chunks.Count; i++)
			{
				// Set size
				sizes[i] = chunks[i].Length;

				// Allocate memory for chunk
				buffers[i] = new CudaDeviceVariable<T>(chunks[i].Length);

				// Copy data to buffer
				buffers[i].CopyToDevice(chunks[i]);

				// Get pointer
				pointers[i] = buffers[i].DevicePointer;

				// Log
				if (i % LogInterval == 0 && !silent)
				{
					this.Log("Data pushed to device", "Chunk: " + i + " / " + chunks.Count, 2, true);
				}
			}

			// Get index pointer
			long indexPointer = buffers.FirstOrDefault()?.DevicePointer.Pointer ?? 0;
			if (indexPointer == 0 && !silent)
			{
				this.Log("Failed to get index pointer", "Ptr: " + indexPointer, 1);
				return 0;
			}

			// Add to Buffers
			Buffers.Add(pointers, sizes);

			// Log
			if (!silent)
			{
				this.Log("Data pushed to device", "IndexPtr: " + indexPointer, 1, true);
			}

			// Return index pointer
			return indexPointer;
		}

		public long PushData<T>(List<T[]> chunks, bool silent = false) where T : unmanaged
		{
			// Create CudaDeviceVariable array & int[]
			CudaDeviceVariable<T>[] buffers = new CudaDeviceVariable<T>[chunks.Count];
			int[] sizes = new int[chunks.Count];
			CUdeviceptr[] pointers = new CUdeviceptr[chunks.Count];

			// Pre log
			if (!silent)
			{
				this.Log("Pushing data to device", "Chunks.Count: " + chunks.Count, 1);
				this.Log("");
			}
			
			// Allocate memory for every chunk
			for (int i = 0; i < chunks.Count; i++)
			{
				// Set size
				sizes[i] = chunks[i].Length;
			
				// Allocate memory for chunk
				buffers[i] = new CudaDeviceVariable<T>(chunks[i].Length);
				
				// Copy data to buffer
				buffers[i].CopyToDevice(chunks[i]);
				
				// Get pointer
				pointers[i] = buffers[i].DevicePointer;
				
				// Log
				if (i % LogInterval == 0 && !silent)
				{
					this.Log("Data pushed to device", "Chunk: " + i + " / " + chunks.Count, 2, true);
				}
			}
			
			// Get index pointer
			long indexPointer = buffers.FirstOrDefault()?.DevicePointer.Pointer ?? 0;
			if (indexPointer == 0 && !silent)
			{
				this.Log("Failed to get index pointer", "Ptr: " + indexPointer, 1);
				return 0;
			}
			
			// Add to Buffers
			Buffers.Add(pointers, sizes);

			// Log
			if (!silent)
			{
				this.Log("Data pushed to device", "IndexPtr: " + indexPointer, 1, true);
			}
			
			// Return index pointer
			return indexPointer;
		}

		public List<T[]> PullData<T>(long pointer, bool aggregate = false, bool silent = false) where T : unmanaged
		{
			// Lookup pointer
			if (!IndexPointers.Contains(pointer) && !silent)
			{
				this.Log("Pointer not found", "Ptr: " + pointer, 1);
				return [];
			}

			// Get pointers from index pointer
			var pointers = GetPointersFromIndex(pointer, out int[] sizes);

			// Abort if no pointers
			if (pointers.Length == 0 && !silent)
			{
				this.Log("No pointers found for index pointer", "Ptr: " + pointer, 1);
				return [];
			}

			// Create List for data
			List<T[]> data = [];

			// Pre log
			if (!silent)
			{
				this.Log("Pulling data from device", "Pointers.Length: " + pointers.Length, 1);
				this.Log("");


				// Pull data from every pointer
				for (int i = 0; i < pointers.Length; i++)
				{
					// Get buffer from pointer
					var buf = pointers[i];

					// Copy data from buffer
					T[] chunk = new T[sizes[i]];
					Ctx.CopyToHost(chunk, buf);

					// Add chunk to data
					data.Add(chunk);

					// Log
					if (i % LogInterval == 0 && !silent)
					{
						this.Log("Data pulled from device", "Chunk: " + i + " / " + pointers.Length, 2, true);
					}
				}

				// Aggregate data
				if (aggregate)
				{
					T[] result = data.SelectMany(x => x).ToArray();

					// Log
					if (!silent)
					{
						this.Log("Data pulled from device", "Length: " + result.Length, 1);
					}

					// Return data
					return [result];
				}

				// Log
				if (!silent)
				{
					this.Log("Data pulled from device", "Count: " + data.Count + " ,Total Length: " + data.Sum(x => x.Length), 1);
				}
			}

			// Return data
			return data;
		}
	}
}