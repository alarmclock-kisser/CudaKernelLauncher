using NAudio.Dmo;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CudaKernelLauncher
{
	public class GuiBuilder
	{
		// ----- ATTRIBUTES ----- \\
		private Panel ParamsPanel;
		private ListBox LogBox;

		public List<NumericUpDown> ParamNumerics = [];
		public List<Label> ParamLabels = [];
		public List<ToolTip> ParamTips = [];

		Dictionary<string, Type> Parameters = [];


		// ----- OBJECTS ----- \\
		private WindowMain Win;




		// ----- LAMBDA ----- \\





		// ----- CONSTRUCTOR ----- \\
		public GuiBuilder(WindowMain window)
		{
			// Set attributes
			Win = window;

			// Find params panel
			ParamsPanel = (FindControl("panel_kernelParams") as Panel) ?? new Panel();
			ParamsPanel.AutoScroll = true;

			// Find log box
			LogBox = (FindControl("listBox_log") as ListBox) ?? new ListBox();
		}




		// ----- METHODS ----- \\
		public void Log(string message, string inner = "", int layer = 1, bool update = false)
		{
			// Abort if no LogBox
			if (LogBox == null)
			{
				return;
			}

			string msg = "[" + DateTime.Now.ToString("HH:mm:ss.fff") + "] ";
			msg += "<GUI>";

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


		public Control? FindControl(string searchName)
		{
			// Find control by name
			Control? found = null;
			foreach (Control control in Win.Controls)
			{
				if (control.Name.ToLower().Contains(searchName.ToLower()))
				{
					found = control;

					if (control.Name == searchName)
					{
						// Found exact match
						Log("Found control exact match: " + control.Name, "", 1, true);
						break;
					}
					else
					{
						// Found partial match
						Log("Found control: " + control.Name, "", 1, true);
					}
				}
			}

			// Return found control
			return found;
		}

		public int BuildParams(Dictionary<string, Type> parameters, int margin = 5, int height = 23, float labelRelativeWidth = 0.5f)
		{
			// Set parameters
			Parameters = parameters;

			// Clear previous params & panel
			ParamsPanel.Controls.Clear();
			ParamNumerics.Clear();
			ParamLabels.Clear();
			ParamTips.Clear();

			// Calculate sizes
			int panelWidth = ParamsPanel.Width;
			int panelHeight = ParamsPanel.Height;
			int labelsWidth = (int) ((panelWidth) * labelRelativeWidth) - (margin * 2);
			int numericsWidth = (int) ((panelWidth) * (1.0f - labelRelativeWidth)) - (margin * 2);

			// Remove pointer (where key is uppercase)
			var pointerParameters = Parameters.Where(x => x.Key == x.Key.ToUpper()).ToDictionary(x => x.Key, x => x.Value);
			var filteredParameters = Parameters.Where(x => x.Key != x.Key.ToUpper()).ToDictionary(x => x.Key, x => x.Value);

			// Remove first int or long
			Dictionary<string, Type>  lengthsParameters = [];
			foreach (var param in filteredParameters.ToList())
			{
				if (param.Value == typeof(int) || param.Value == typeof(long))
				{
					lengthsParameters.Add(param.Key, param.Value);
					filteredParameters.Remove(param.Key);
				}
				if (lengthsParameters.Count >= pointerParameters.Count)
				{
					break;
				}
			}

			// DEBUG log parameters
			Log("Parameters: " + filteredParameters.Count + " (pointers: " + pointerParameters.Count + ", lengths: " + lengthsParameters.Count + ")");
			foreach (var param in filteredParameters)
			{
				Log("Param: " + param.Key + " (" + param.Value.Name + ")");
			}

			// DEBUG log sizes
			Log("Panel dimensions: " + panelWidth + "px width x " + panelHeight + "px height");
			Log("Margin : " + margin + "px, Height: " + height + "px");
			Log("Label width: " + labelsWidth + "px", "Relative: " + (int) (labelRelativeWidth * 100) + "%");
			Log("Numeric width: " + numericsWidth + "px", "Relative: " + (int) ((1.0f - labelRelativeWidth) * 100) + "%");

			// Build label at first position with pointer and length
			Label mandatoryInfo = new Label
				{
					Name = "label_mandatoryParamsInfo",
					Text = "Pointer(s): (" + pointerParameters.Count + ") " + string.Join(", ", pointerParameters.Keys) + ", Length(s): (" + lengthsParameters.Count + ") " + string.Join(", ", lengthsParameters.Keys),
				AutoSize = true,
					Location = new Point(margin, margin),
					Width = panelWidth - (margin * 2),
					Height = height
				};
			ParamsPanel.Controls.Add(mandatoryInfo);

			// Build params into panel: margin + Label + margin + margin + NumericUpDown + margin
			int yOff = margin + height + margin + margin; // Start at first label position
			for (int i = 0; i < filteredParameters.Count; i++)
			{
				// Get init. attributes for numeric
				Type t = filteredParameters.ElementAt(i).Value;
				decimal min = -1;
				decimal max = 1;
				decimal inc = 1;
				int decimals = 0;
				decimal value = 0;
				switch(t.Name)
				{
					case "Int32":
						min = int.MinValue;
						max = int.MaxValue;
						inc = 1;
						decimals = 0;
						value = 0;
						break;
					case "Int64":
						min = long.MinValue;
						max = long.MaxValue;
						inc = 1;
						decimals = 0;
						value = 100;
						break;
					case "Single":
						min = -5.0M;
						max = 10.0M;
						inc = 0.01m;
						decimals = 6;
						value = 1.0M;
						break;
					case "Double":
						min = -16384;
						max = 16384;
						inc = 0.0001m;
						decimals = 12;
						value = 0.5M;
						break;
					case "Decimal":
						min = -16384;
						max = 16384;
						inc = 0.000001m;
						decimals = 22;
						value = 0.5M;
						break;
					case "Byte":
						min = byte.MinValue;
						max = byte.MaxValue;
						inc = 1;
						decimals = 0;
						value = 128;
						break;
					default:
						Log("Unknown type: " + t.Name, "", 1);
						continue; // Skip unknown types
				}

				// Create label
				Label label = new Label
				{
					Name = "label_param_" + filteredParameters.ElementAt(i).Key,
					Text = filteredParameters.ElementAt(i).Key,
					AutoSize = true,
					Location = new Point(margin, yOff),
					Width = labelsWidth,
					Height = height
				};

				// Create numeric
				NumericUpDown numeric = new NumericUpDown
				{
					Name = "numeric_param_" + filteredParameters.ElementAt(i).Key,
					Location = new Point(labelsWidth + (margin * 2), yOff),
					Width = numericsWidth,
					Height = height,
					Minimum = min,
					Maximum = max,
					Increment = inc,
					DecimalPlaces = decimals,
					Value = value,
					Hexadecimal = (t.Name.ToLower() == "byte"),
				};

				// Create tooltip on label
				ToolTip tip = new ToolTip();
				tip.SetToolTip(label, filteredParameters.ElementAt(i).Value.Name);

				// Add to panel
				ParamsPanel.Controls.Add(label);
				ParamsPanel.Controls.Add(numeric);
				
				// Add to lists
				ParamLabels.Add(label);
				ParamNumerics.Add(numeric);
				
				// Increment offset
				yOff += height + (margin * 2);
			}

			return filteredParameters.Count;
		}

		public object[] GetParamValues()
		{
			// Hole Parameter-Typen
			var paramTypes = Parameters.Values.ToList();

			// Hole Werte aus ParamNumerics
			object[] values = new object[ParamNumerics.Count];

			for (int i = 0; i < ParamNumerics.Count; i++)
			{
				decimal value = ParamNumerics[i].Value;
				Type type = paramTypes[i];

				// Konvertiere Wert in den richtigen Typ
				values[i] = type == typeof(int) ? (object) (int) value :
							type == typeof(long) ? (long) value :
							type == typeof(float) ? (float) value :
							type == typeof(double) ? (double) value :
							type == typeof(decimal) ? (decimal) value :
							type == typeof(byte) ? (byte) value :
							throw new InvalidOperationException($"Nicht unterstützter Typ: {type}");
			}

			return values;
		}


	}




}
