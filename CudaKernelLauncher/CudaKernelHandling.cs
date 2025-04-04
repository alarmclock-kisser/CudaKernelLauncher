using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.NVRTC;
using ManagedCuda.VectorTypes;
using System.Diagnostics;

namespace CudaKernelLauncher
{
	public class CudaKernelHandling
	{
		// ----- ATTRIBUTES ----- \\
		private CudaContextHandling CudaH;
		private PrimaryContext Ctx;

		public CudaKernel? Kernel = null;


		// Mapping für primitive Typen
		private static readonly Dictionary<string, Type> PrimitiveTypes = new()
		{
			{ "int", typeof(int) },
			{ "float", typeof(float) },
			{ "double", typeof(double) },
			{ "bool", typeof(bool) },
			{ "string", typeof(string) },
			{ "char", typeof(char) },
			{ "byte", typeof(byte) },
			{"short", typeof(short) },
			{ "long", typeof(long) },
			{ "uint", typeof(uint) },
			{"ulong", typeof(ulong) },
			{ "ushort", typeof(ushort) },
			{ "sbyte", typeof(sbyte) },
			{ "object", typeof(object) }
		};


		// ----- LAMBDA ----- \\
		public string Repopath => CudaH.Repopath;


		private CudaMemoryHandling? MemH => CudaH.MemH;


		public string[] KernelPaths => GetKernelPaths();

		public string? KernelName => Kernel?.KernelName ?? null;
		public Dictionary<string, Type>? KernelParameters => Kernel != null ? GetKernelParameters(KernelString ?? "", true) : null;
		public string? KernelString => Kernel != null ? GetKernelString(Kernel?.KernelName ?? "") : null;


		public CudaKernel? this[string kernelName] => LoadKernelByName(kernelName, true);
		public string this[int index] => !(index < 0 || index >= KernelPaths.Length) ? KernelPaths[index] : "";

		private int LogInterval => CudaH.LogInterval;


		// ----- CONSTRUCTOR ----- \\
		public CudaKernelHandling(CudaContextHandling cudaHandling, PrimaryContext ctx)
		{
			this.CudaH = cudaHandling;
			this.Ctx = ctx;

			// Verify Kernel folder structure
			VerifyFolderStructure();
		}



		// ----- METHODS ----- \\
		// Log
		public void Log(string message, string inner = "", int level = 1, bool update = false)
		{
			string msg = "[" + DateTime.Now.ToString("hh:mm:ss.fff") + "] ";
			msg += "<Kernel> ";
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


		// I/O
		public void VerifyFolderStructure(bool silent = false)
		{
			// Check for Dir path: Resources\Kernels\PTX & Resources\Kernels\CU , if not fully available, create them
			string ptxPath = Path.Combine(Repopath, "Kernels\\PTX");
			string cuPath = Path.Combine(Repopath, "Kernels\\CU");

			if (!Directory.Exists(ptxPath))
			{
				Directory.CreateDirectory(ptxPath);
			}
			if (!Directory.Exists(cuPath))
			{
				Directory.CreateDirectory(cuPath);
			}

			if (!silent)
			{
				Log("Folder structure verified", "", 1);
			}
		}

		public string[] GetKernelPaths()
		{
			string ptxPath = Path.Combine(Repopath, "Kernels\\PTX");
			string[] ptxFiles = Directory.GetFiles(ptxPath, "*.ptx");
			return ptxFiles.ToArray();
		}

		public string? GetKernelString(string kernelName)
		{
			// Verify kernel name ends with "Kernel"
			kernelName = kernelName.Replace("Kernel", "");
			kernelName += "Kernel";

			// Check in \CU folder for .c file with kernelName
			string cPath = Path.Combine(Repopath, "Kernels\\CU", kernelName + ".c");
			if (File.Exists(cPath))
			{
				return File.ReadAllText(cPath);
			}

			// Check for .cu file with kernelName
			string cuPath = Path.Combine(Repopath, "Kernels\\CU", kernelName + ".cu");
			if (File.Exists(cuPath))
			{
				return File.ReadAllText(cuPath);
			}

			return null;
		}

		public Dictionary<string, Type> GetKernelParameters(string? kernelString = null, bool silent = false)
		{
			kernelString ??= KernelString;
			if (kernelString == null)
			{
				if (!silent)
				{
					Log("No kernel string provided", "", 1);
				}
				return [];
			}

			// New dict. for parameters
			Dictionary<string, Type> parametersDict = [];

			// Precompile kernel string, check result (name)
			string? kernelName = PrecompileKernelString(kernelString, true);
			if (kernelName == null)
			{
				if (!silent)
				{
					Log("Failed to precompile kernel string", "", 1);
				}
				return parametersDict;
			}

			// Split in lines
			string[] lines = kernelString.Split('\n');

			// Get line index  with "__global__ void "
			int index = Array.FindIndex(lines, x => x.Contains("void " + kernelName));
			if (index == -1)
			{
				if (!silent)
				{
					Log("Fatal error: 'void '" + kernelName + " not found", "This should have been prevented by PrecompileKernelString()", 2);
				}
				return parametersDict;
			}
			string paramsLine = lines[index];

			// Get parameters (between "(" and ")")
			paramsLine = paramsLine.Split("(")[1].Split(")")[0];

			// Split parameters, trim each
			string[] parameters = paramsLine.Split(",").Select(x => x.Trim()).ToArray();

			// Get name and type for each parameter
			for (int i = 0; i < parameters.Length; i++)
			{
				string param = parameters[i].Trim();

				// Split by space
				string[] parts = param.Split(" ");

				// Get name and type
				string name = parts.LastOrDefault() ?? "param" + i + 1;
				string type = parts.FirstOrDefault() ?? "none";

				// If type has * remove it and put name in uppercase
				if (type.Contains("*"))
				{
					type = type.Replace("*", "");
					name = name.ToUpper();

					if (!silent)
					{
						Log("Pointer as parameter detected", "Type: " + type + ", Name: " + name, 2);
					}
				}
				else
				{
					if (!silent)
					{
						Log("Parameter detected", "Type: " + type + ", Name: " + name, 2);
					}
				}

				// Typ abrufen mit Fallback
				Type t = PrimitiveTypes.TryGetValue(type, out Type? foundType)
					? foundType
					: Type.GetType(type) ?? typeof(object);

				// Add to dict
				parametersDict.Add(name, t);
			}

			// Return parameters
			return parametersDict;
		}

		public int GetStretchKernelsCount()
		{
			string[] kernelPaths = GetKernelPaths();
			return kernelPaths.Count(x => x.Contains("Stretch"));
		}

		public string GetStretchKernelName(int index, bool silent = false)
		{
			string[] kernelPaths = GetKernelPaths();
			string[] stretchKernels = kernelPaths.Where(x => x.Contains("TimeStretch")).ToArray();

			// Verify index range
			if (index < 0 || index >= stretchKernels.Length)
			{
				if (!silent)
				{
					Log("Index out of range", "Index: " + index + ", Count: " + stretchKernels.Length, 1);
				}
				return "";
			}

			return stretchKernels[index];
		}



		// Compile
		public string? PrecompileKernelString(string kernelString, bool silent = false)
		{
			// Check contains "extern c"
			if (!kernelString.Contains("extern \"C\""))
			{
				if (!silent)
				{
					Log("Kernel string does not contain 'extern \"C\"'", "", 1);
				}
				return null;
			}

			// Check contains "__global__ "
			if (!kernelString.Contains("__global__"))
			{
				if (!silent)
				{
					Log("Kernel string does not contain '__global__'", "", 1);
				}
				return null;
			}

			// Check contains "void "
			if (!kernelString.Contains("void "))
			{
				if (!silent)
				{
					Log("Kernel string does not contain 'void '", "", 1);
				}
				return null;
			}

			// Check contains int
			if (!kernelString.Contains("int ") && !kernelString.Contains("long "))
			{
				if (!silent)
				{
					Log("Kernel string does not contain 'int ' (for array length)", "", 1);
				}
				return null;
			}

			// Check if every bracket is closed (even amount) for {} and () and []
			int open = kernelString.Count(c => c == '{');
			int close = kernelString.Count(c => c == '}');
			if (open != close)
			{
				if (!silent)
				{
					Log("Kernel string has unbalanced brackets", " { } ", 1);
				}
				return null;
			}
			open = kernelString.Count(c => c == '(');
			close = kernelString.Count(c => c == ')');
			if (open != close)
			{
				if (!silent)
				{
					Log("Kernel string has unbalanced brackets", " ( ) ", 1);
				}
				return null;
			}
			open = kernelString.Count(c => c == '[');
			close = kernelString.Count(c => c == ']');
			if (open != close)
			{
				if (!silent)
				{
					Log("Kernel string has unbalanced brackets", " [ ] ", 1);
				}
				return null;
			}

			// Check if kernel contains "blockIdx.x" and "blockDim.x" and "threadIdx.x"
			if (!kernelString.Contains("blockIdx.x") || !kernelString.Contains("blockDim.x") || !kernelString.Contains("threadIdx.x"))
			{
				if (!silent)
				{
					Log("Kernel string should contain 'blockIdx.x', 'blockDim.x' and 'threadIdx.x'", "", 2);
				}
			}

			// Get name between "void " and "("
			int start = kernelString.IndexOf("void ") + "void ".Length;
			int end = kernelString.IndexOf("(", start);
			string name = kernelString.Substring(start, end - start);

			// Trim every line ends from empty spaces (split -> trim -> aggregate)
			kernelString = kernelString.Split("\n").Select(x => x.TrimEnd()).Aggregate((x, y) => x + "\n" + y);

			// Log name
			if (!silent)
			{
				Log("Succesfully precompiled kernel string", "Name: " + name, 1);
			}

			return name;
		}

		public string CompileKernel(string filepath)
		{
			if (Ctx == null)
			{
				Log("No CUDA context available", "", 1);
				return "";
			}

			// If file is not a .cu file, but raw kernel string, compile that
			if (Path.GetExtension(filepath) != ".cu")
			{
				return CompileString(filepath);
			}

			string kernelName = Path.GetFileNameWithoutExtension(filepath);
			kernelName = kernelName.Replace("Kernel", "");

			string logpath = Path.Combine(Repopath, "Logs", kernelName + "Kernel" + ".log");

			Stopwatch sw = Stopwatch.StartNew();
			Log("Compiling kernel " + kernelName);

			// Load kernel file
			string kernelCode = File.ReadAllText(filepath);


			var rtc = new CudaRuntimeCompiler(kernelCode, kernelName);

			try
			{
				// Compile kernel
				rtc.Compile([]);

				if (rtc.GetLogAsString().Length > 0)
				{
					Log("Kernel compiled with warnings", "", 1);
					File.WriteAllText(logpath, rtc.GetLogAsString());
				}


				sw.Stop();
				long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
				Log("Kernel compiled within " + deltaMicros.ToString("N0") + " µs", "Repo\\" + Path.GetRelativePath(Repopath, logpath), 2, true);

				// Get ptx code
				byte[] ptxCode = rtc.GetPTX();

				// Export ptx
				string ptxPath = Path.Combine(Repopath, "Kernels\\PTX", kernelName + "Kernel" + ".ptx");
				File.WriteAllBytes(ptxPath, ptxCode);

				Log("PTX code exported to " + ptxPath, "", 1);

				return ptxPath;
			}
			catch (Exception ex)
			{
				File.WriteAllText(logpath, rtc.GetLogAsString());
				Log(ex.Message, ex.InnerException?.Message ?? "", 1);

				return "";
			}

		}

		public string CompileString(string kernelString)
		{
			if (Ctx == null)
			{
				Log("No CUDA context available", "", 1);
				return "";
			}

			string kernelName = kernelString.Split("void ")[1].Split("(")[0];
			kernelName = kernelName.Replace("Kernel", "");

			string logpath = Path.Combine(Repopath, "Logs", kernelName + "Kernel" + ".log");

			Stopwatch sw = Stopwatch.StartNew();
			Log("Compiling kernel " + kernelName);

			// Load kernel file
			string kernelCode = kernelString;

			// Save also the kernel string as .c file
			string cPath = Path.Combine(Repopath, "Kernels\\CU", kernelName + "Kernel" + ".c");
			File.WriteAllText(cPath, kernelCode);


			var rtc = new CudaRuntimeCompiler(kernelCode, kernelName);

			try
			{
				// Compile kernel
				rtc.Compile([]);

				if (rtc.GetLogAsString().Length > 0)
				{
					Log("Kernel compiled with warnings", "", 1);
					File.WriteAllText(logpath, rtc.GetLogAsString());
				}


				sw.Stop();
				long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
				Log("Kernel compiled in " + deltaMicros.ToString("N0") + " µs", "Repo\\" + Path.GetRelativePath(Repopath, logpath), 2, true);

				// Get ptx code
				byte[] ptxCode = rtc.GetPTX();

				// Export ptx
				string ptxPath = Path.Combine(Repopath, "Kernels\\PTX", kernelName + "Kernel" + ".ptx");
				File.WriteAllBytes(ptxPath, ptxCode);

				Log("PTX code exported to " + ptxPath, "", 1);

				return ptxPath;
			}
			catch (Exception ex)
			{
				File.WriteAllText(logpath, rtc.GetLogAsString());
				Log(ex.Message, ex.InnerException?.Message ?? "", 1);

				return "";
			}
		}


		// Load
		public CudaKernel? LoadKernel(string filepath, bool silent = false)
		{
			if (Ctx == null)
			{
				Log("No CUDA context available", "", 1);
				return null;
			}

			// Unload?
			if (Kernel != null)
			{
				UnloadKernel(true);
			}

			// Get kernel name
			string kernelName = Path.GetFileNameWithoutExtension(filepath);
			kernelName = kernelName.Replace("Kernel", "");

			// Get log path
			string logpath = Path.Combine(Repopath, "Logs", kernelName + "Kernel" + ".log");

			// Log
			Stopwatch sw = Stopwatch.StartNew();
			if (!silent)
			{
				Log("Started loading kernel " + kernelName);
			}

			// Try to load kernel
			try
			{
				// Load ptx code
				byte[] ptxCode = File.ReadAllBytes(filepath);

				// Load kernel
				Kernel = Ctx.LoadKernelPTX(ptxCode, kernelName);
			}
			catch (Exception ex)
			{
				if (!silent)
				{
					Log("Failed to load kernel " + kernelName, ex.Message, 1);
					string logMsg = ex.Message + Environment.NewLine + Environment.NewLine + ex.InnerException?.Message ?? "";
					File.WriteAllText(logpath, logMsg);
				}
				Kernel = null;
			}

			// Log
			sw.Stop();
			long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
			if (!silent)
			{
				Log("Kernel loaded within " + deltaMicros.ToString("N0") + " µs", "", 2);
			}

			return Kernel;
		}

		public CudaKernel? LoadKernelByName(string kernelName, bool silent = false)
		{
			if (Ctx == null)
			{
				Log("No CUDA context available", "", 1);
				return null;
			}

			// Abort if no kernel name
			if (kernelName == "")
			{
				if (!silent)
				{
					Log("No kernel name provided", "", 1);
				}
				return null;
			}

			// Get kernel path
			string kernelPath = Path.Combine(Repopath, "Kernels\\PTX", kernelName.Replace("Kernel", "") + "Kernel" + ".ptx");

			return LoadKernel(kernelPath, silent);
		}

		public void UnloadKernel(bool silent = false)
		{
			if (Ctx == null)
			{
				Log("No CUDA context available", "", 1);
				return;
			}


			Kernel = null;

			if (!silent)
			{
				Log("Kernel unloaded", "", 1);
			}
		}


		// Dispose
		public void Dispose()
		{
			UnloadKernel(true);
		}


		// Execute
		public void ExecuteKernelOld(long indexPointer, float param1, float? param2 = null)
		{
			if (Ctx == null || MemH == null)
			{
				Log("No CUDA context available", "", 1);
				return;
			}
			if (Kernel == null)
			{
				Log("No kernel loaded", "", 1);
				return;
			}
			if (!MemH.IndexPointers.Contains(indexPointer))
			{
				Log("No input variables found", "", 1);
				return;
			}

			// Get input variables
			CUdeviceptr[] pointers = MemH.GetPointersFromIndex(indexPointer, out int[] sizes);

			// Set grid and block size
			Kernel.BlockDimensions = new dim3(256, 1, 1);
			Kernel.GridDimensions = new dim3((int) Math.Ceiling(pointers.LongLength / 256.0), 1, 1);

			// Run kernel for each pointer
			Stopwatch sw = Stopwatch.StartNew();
			Log("Started running kernel " + Kernel.KernelName + " on " + pointers.LongLength + " input variables");
			Log("");

			// Run for each pointer
			for (int i = 0; i < pointers.LongLength; i++)
			{
				if (param2 == null)
				{
					// Single parameter
					Kernel.Run(pointers[i], sizes[i], param1);
				}
				else
				{
					// Double parameter
					Kernel.Run(pointers[i], sizes[i], param1, param2.Value);
				}

				// Log progress
				if (i % LogInterval == 0)
				{
					Log("Ran kernel on input " + i + " / " + pointers.LongLength, "", 2, true);
				}
			}

			// Log final
			sw.Stop();
			long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
			Log("Ran kernel on " + pointers.LongLength + " pointers within " + deltaMicros.ToString("N0") + " µs", "Ptr: " + indexPointer, 1, true);
		}

		public long ExecuteKernel(long indexPointer, params object[] parameters)
		{
			// Abort if components fail
			if (Ctx == null || MemH == null)
			{
				Log("No CUDA context available", "", 1);
				return 0;
			}
			if (Kernel == null || KernelParameters == null)
			{
				Log("No kernel loaded", "", 1);
				return indexPointer;
			}
			if (!MemH.IndexPointers.Contains(indexPointer))
			{
				Log("No input variables found", "", 1);
				return 0;
			}
			if (parameters.Length != KernelParameters.Count - 2)
			{
				Log("Parameter count mismatch", "Expected: (" + KernelParameters.Count + " - 2)" + ", Got: " + parameters.Length, 1);
				return indexPointer;
			}

			// Get parameters as string
			string paramStr = "";
			foreach (object param in parameters)
			{
				paramStr += param + ", ";
			}

			// Get input variables
			CUdeviceptr[] pointers = MemH.GetPointersFromIndex(indexPointer, out int[] sizes);

			// Set grid and block size
			Kernel.BlockDimensions = new dim3(256, 1, 1);
			Kernel.GridDimensions = new dim3((int) Math.Ceiling(pointers.LongLength / 256.0), 1, 1);

			// Log & stopwatch
			Stopwatch sw = Stopwatch.StartNew();
			Log("Started running kernel " + Kernel.KernelName + " on " + pointers.LongLength + " input variables", "Arguments: " + paramStr);
			Log("");

			// Run for each pointer
			for (int n = 0; n < pointers.LongLength; n++)
			{
				// Typen korrekt extrahieren
				List<object> formattedParams = [];

				foreach (object param in parameters)
				{
					switch (param)
					{
						case int i: formattedParams.Add(i); break;
						case float f: formattedParams.Add(f); break;
						case decimal d: formattedParams.Add((float) d); break; // CUDA mag kein decimal
						case CUdeviceptr ptr: formattedParams.Add(ptr); break; // Falls GPU-Pointer
						default: throw new ArgumentException($"Unsupported CUDA parameter type: {param.GetType()}");
					}
				}

				// Run kernel with parameters
				if (KernelParameters.ElementAt(0).Value == typeof(int))
				{
					// Size first argument, second pointer, rest parameters
					Kernel.Run([sizes[n], pointers[n], .. formattedParams]);
				}
				else
				{
					// Pointer first argument, second size, rest parameters
					Kernel.Run([pointers[n], sizes[n], .. formattedParams]);
				}

				// Log progress
				if (n % LogInterval == 0)
				{
					Log("Ran kernel on input " + n + " / " + pointers.LongLength, "", 2, true);
				}
			}

			// Log final
			sw.Stop();
			long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
			Log("Ran kernel on " + pointers.LongLength + " pointers within " + deltaMicros.ToString("N0") + " µs", "Ptr: " + indexPointer, 1, true);

			// Return index pointer
			return indexPointer;
		}

		public long ExecuteKernelOOP(long indexPointer, params object[] parameters)
		{
			// Abort if components fail
			if (Ctx == null || MemH == null)
			{
				Log("No CUDA context available", "", 1);
				return 0;
			}
			if (Kernel == null || KernelParameters == null)
			{
				Log("No kernel loaded", "", 1);
				return indexPointer;
			}
			if (!MemH.IndexPointers.Contains(indexPointer))
			{
				Log("No input variables found", "", 1);
				return 0;
			}
			if (parameters.Length != KernelParameters.Count - 2)
			{
				Log("Parameter count mismatch", "Expected: (" + KernelParameters.Count + " - 2)" + ", Got: " + parameters.Length, 1);
				return indexPointer;
			}

			// Get parameters as string
			string paramStr = "";
			foreach (object param in parameters)
			{
				paramStr += param + ", ";
			}

			// Get input variables
			CUdeviceptr[] pointers = MemH.GetPointersFromIndex(indexPointer, out int[] sizes);

			// Set grid and block size
			Kernel.BlockDimensions = new dim3(256, 1, 1);
			Kernel.GridDimensions = new dim3((int) Math.Ceiling(pointers.LongLength / 256.0), 1, 1);

			// Log & stopwatch
			Stopwatch sw = Stopwatch.StartNew();
			Log("Started running kernel " + Kernel.KernelName + " on " + pointers.LongLength + " input variables", "Arguments: " + paramStr);
			Log("");

			// Run for each pointer
			for (int n = 0; n < pointers.LongLength; n++)
			{
				// Typen korrekt extrahieren
				List<object> formattedParams = [];

				foreach (object param in parameters)
				{
					switch (param)
					{
						case int i: formattedParams.Add(i); break;
						case float f: formattedParams.Add(f); break;
						case decimal d: formattedParams.Add((float) d); break; // CUDA mag kein decimal
						case CUdeviceptr ptr: formattedParams.Add(ptr); break; // Falls GPU-Pointer
						default: throw new ArgumentException($"Unsupported CUDA parameter type: {param.GetType()}");
					}
				}

				// Run kernel with parameters
				if (KernelParameters.ElementAt(0).Value == typeof(int))
				{
					// Size first argument, second pointer, rest parameters
					Kernel.Run([sizes[n], pointers[n], .. formattedParams]);
				}
				else
				{
					// Pointer first argument, second size, rest parameters
					Kernel.Run([pointers[n], sizes[n], .. formattedParams]);
				}

				// Log progress
				if (n % LogInterval == 0)
				{
					Log("Ran kernel on input " + n + " / " + pointers.LongLength, "", 2, true);
				}
			}

			// Log final
			sw.Stop();
			long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
			Log("Ran kernel on " + pointers.LongLength + " pointers within " + deltaMicros.ToString("N0") + " µs", "Ptr: " + indexPointer, 1, true);

			// Return index pointer
			return indexPointer;
		}


	}
}