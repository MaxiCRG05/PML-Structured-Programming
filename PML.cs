using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Perceptron_Multicapa_Colores
{
	/// <summary>
	/// Clase del Perceptrón MultiCapa
	/// </summary>
	class PML
	{
		/// <summary>
		/// capas: Variable para obtener las capas.
		/// </summary>
		private readonly int[] capas;

		/// <summary>
		/// delta: Variable para obtener el delta.
		/// </summary>
		private readonly double[][] delta;

		/// <summary>
		/// Instancia de la clase Archivos.
		/// </summary>
		public readonly Archivos archivo;

		/// <summary>
		/// neuronas: Variable para obtener las neuronas.
		/// </summary>
		private double[][] neuronas;

		/// <summary>
		/// pesos: Variable para obtener los pesos.
		/// </summary>
		private double[][][] pesos;

		/// <summary>
		/// bias: Variable para obtener los sesgos.
		/// </summary>
		private double[][] bias;

		/// <summary>
		/// Constructor de la clase PML.
		/// </summary>
		/// <param name="layers"></param>
		public PML(int[] layers)
		{
			capas = layers;
			delta = new double[capas.Length][];
			archivo = new Archivos(VariablesGlobales.Ruta);
			IniciarNeuronas();
			IniciarBias();
			IniciarPesos();
		}

		/// <summary>
		/// Método para iniciar las neuronas.
		/// </summary>
		private void IniciarNeuronas()
		{
			neuronas = new double[capas.Length][];

			for (int i = 0; i < capas.Length; i++)
			{
				neuronas[i] = new double[capas[i]];
			}
		}

		/// <summary>
		/// MÉTODO DE XAVIER (GLOROT)
		/// Método para iniciar los pesos. 
		/// </summary>
		private void IniciarPesos()
		{
			Random rand = new Random();
			pesos = new double[capas.Length - 1][][];
			for (int i = 0; i < capas.Length - 1; i++)
			{
				int n_in = capas[i];
				int n_out = capas[i + 1];
				double limit = (double)Math.Sqrt(6.0 / (n_in + n_out));

				pesos[i] = new double[n_out][];
				for (int j = 0; j < n_out; j++)
				{
					pesos[i][j] = new double[n_in];
					for (int k = 0; k < n_in; k++)
					{
						pesos[i][j][k] = (double)(rand.NextDouble() * 2 * limit - limit);
					}
				}
			}
		}

		/// <summary>
		/// Método para iniciar los sesgos.
		/// </summary>
		private void IniciarBias()
		{
			bias = new double[capas.Length - 1][];
			for (int i = 0; i < capas.Length - 1; i++)
			{
				bias[i] = new double[capas[i + 1]];
				for (int j = 0; j < capas[i + 1]; j++)
				{
					bias[i][j] = (double)new Random().NextDouble() - 0.5f;
				}
			}
		}

		/// <summary>
		/// Método para entrenar la red neuronal.
		/// </summary>
		public void Entrenar()
		{
			double mejorError = double.MaxValue;
			int epocasSinMejora = 0;
			const int paciencia = 1000;

			for (int epoca = 0; epoca < VariablesGlobales.Epocas; epoca++)
			{
				double errorEpoca = 0;

				for (int i = 0; i < VariablesGlobales.Entradas.Length; i++)
				{
					Propagación(VariablesGlobales.Entradas[i]);
					Retropropagación(VariablesGlobales.Salidas[i]);

					for (int j = 0; j < VariablesGlobales.Salidas[i].Length; j++)
					{
						errorEpoca += Math.Pow(VariablesGlobales.Salidas[i][j] - neuronas[capas.Length - 1][j], 2); 
					}
				}	

				errorEpoca /= (VariablesGlobales.Entradas.Length * VariablesGlobales.Salidas[0].Length);

				if (errorEpoca < mejorError)
				{
					mejorError = errorEpoca;
					epocasSinMejora = 0;
				}
				else
				{
					epocasSinMejora++;
					if (epocasSinMejora >= paciencia)
					{
						Console.WriteLine($"Entrenamiento detenido en la época {epoca + 1}. Error: {errorEpoca}");
						break;
					}
				}

				if (errorEpoca <= VariablesGlobales.ErrorMinimo)
				{
					Console.WriteLine($"Entrenamiento detenido en la época {epoca + 1}. El error se disminuyó: {errorEpoca}");
					MessageBox.Show($"Entrenamiento detenido en la época {epoca + 1}. El error se disminuyó: {errorEpoca}", "Entrenamiento");
					break;
				}
			}
			MessageBox.Show($"Entrenamiento finalizado.", "Entrenamiento");
		}

		/// <summary>
		/// Método para la propagación de la red neuronal.
		/// </summary>
		/// <param name="x">Patrones para procesar por la red neuronal.</param>
		/// <returns>Retorna la salida obtenida por la red neuronal.</returns>
		public double[] Propagación(double[] x)
		{
			x = NormalizarValores(x);

			for (int i = 0; i < x.Length; i++)
			{
				neuronas[0][i] = x[i];
			}

			for (int c = 1; c < capas.Length; c++)
			{
				for (int j = 0; j < capas[c]; j++)
				{
					double suma = 0;
					for (int k = 0; k < capas[c - 1]; k++)
					{
						suma += neuronas[c - 1][k] * pesos[c - 1][j][k];
					}
					neuronas[c][j] = FuncionActivacion(suma + bias[c - 1][j]);
				}
			}

			neuronas[capas.Length - 1] = Softmax(neuronas[capas.Length - 1]);

			return neuronas[capas.Length - 1];
		}

		/// <summary>
		/// Método para la retropropagación de la red neuronal.
		/// </summary>
		/// <param name="salidaEsperada">Salidas esperadas por parte de la red neuronal.</param>
		private void Retropropagación(double[] salidaEsperada)
		{
			double[] errores = new double[capas[capas.Length - 1]];

			for (int c = 0; c < capas.Length; c++)
			{
				delta[c] = new double[capas[c]];
			}
				
			for (int i = 0; i < errores.Length; i++)
			{
				double output = neuronas[capas.Length - 1][i];
				errores[i] = (salidaEsperada[i] - output);
				delta[capas.Length - 1][i] = errores[i] * FuncionDeActivacionDerivada(output);
			}

			for (int c = capas.Length - 2; c >= 0; c--)
			{
				for (int j = 0; j < capas[c]; j++)
				{
					double error = 0;
					for (int k = 0; k < capas[c + 1]; k++)
					{
						error += delta[c + 1][k] * pesos[c][k][j];
					}
					delta[c][j] = error * FuncionDeActivacionDerivada(neuronas[c][j]);
				}
			}

			for (int c = 0; c < capas.Length - 1; c++)
			{
				for (int j = 0; j < capas[c + 1]; j++)
				{
					for (int k = 0; k < capas[c]; k++)
					{
						pesos[c][j][k] += VariablesGlobales.TasaAprendizaje * delta[c + 1][j] * neuronas[c][k];
					}
					bias[c][j] += VariablesGlobales.TasaAprendizaje * delta[c + 1][j];
				}
			}
		}

		/// <summary>
		/// Método para la función de activación Softmax.
		/// </summary>
		/// <param name="x">valores de entrada.</param>
		/// <returns>Valores ya procesados.</returns>
		private double[] Softmax(double[] x)
		{
			double[] expValues = x.Select(n => (double)Math.Exp(n)).ToArray();
			double sumExpValues = expValues.Sum();
			return expValues.Select(n => n / sumExpValues).ToArray();
		}

		/// <summary>
		/// Método para la función de activación: FUNCION RELU, LEAKY RELU Y SIGMOIDE
		/// </summary>
		/// <param name="x">Valor de entrada.</param>
		/// <returns>Regresa el valor de entrada procesada con base a la función de activación.</returns>
		private double FuncionActivacion(double x)
		{
			//SIGMOIDE
			//return 1 / (1 + Math.Exp(-x));

			//RELU
			//return Math.Max(0,x);

			// Leaky ReLU
			return x > 0 ? x : 0.01 * x;
		}

		/// <summary>
		/// Método para la función de activación derivada: FUNCION RELU, LEAKY RELU Y SIGMOIDE DERIVADA
		/// </summary>
		/// <param name="x">Valor de entrada.</param>
		/// <returns>La entrada ya procesada por la función de activación derivada.</returns>
		private double FuncionDeActivacionDerivada(double x)
		{
			//SIGMOIDE
			//return x * (1 - x);

			//RELU
			//return x > 0 ? 1 : 0;

			//Leaky ReLU
			return x > 0 ? 1 : 0.01;
		}

		/// <summary>
		/// Método para normalizar los valores.
		/// </summary>
		/// <param name="x">Valor para normalizar</param>
		/// <returns>Regresa el valor normalizado</returns>
		public double NormalizarValores(double x)
		{
			return (x - VariablesGlobales.Min) / (VariablesGlobales.Max - VariablesGlobales.Min);
		}

		/// <summary>
		/// Método para normalizar los valores.
		/// </summary>
		/// <param name="x">Valores de entrada.</param>
		/// <returns>Regresa los valores ya normalizados.</returns>
		/// <exception cref="Exception">El valor excede el rango</exception>
		public double[] NormalizarValores(double[] x)
		{
			double[] resultado = new double[x.Length];

			for (int i = 0; i < x.Length; i++)
			{
				resultado[i] = (x[i] - VariablesGlobales.Min) / (VariablesGlobales.Max - VariablesGlobales.Min);
				if (resultado[i] < 0 || resultado[i] > 1)
				{
					throw new Exception($"Valor normalizado fuera de rango: {resultado[i]}");
				}
			}

			return resultado;
		}

		/// <summary>
		/// Método para cargar los datos.
		/// </summary>
		/// <returns>Regresa "true" si se pueden cargar los datos.</returns>
		public bool CargarDatos()
		{
			try
			{
				string line;
				int capaActual = -1;
				int neuronaActual = 0;
				int pesoActual = 0;

				while ((line = archivo.LeerArchivo(VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos)) != null)
				{
					if (line.StartsWith("Capa"))
					{
						capaActual++;
						neuronaActual = 0;
						pesoActual = 0;
					}
					else if (line.StartsWith("Peso"))
					{
						string[] partes = line.Split('=');
						double peso = double.Parse(partes[1].Trim());


						pesoActual++;
						if (pesoActual >= capas[capaActual])
						{
							pesoActual = 0;
							neuronaActual++;
						}
					}
					else if (line.StartsWith("Sesgo"))
					{
						string[] partes = line.Split('=');
						double sesgo = double.Parse(partes[1].Trim());

						if (capaActual >= 0 && capaActual < bias.Length && neuronaActual < bias[capaActual].Length)
						{
							bias[capaActual][neuronaActual] = sesgo;
							neuronaActual++;
						}
					}
				}
				
				MessageBox.Show("Configuración cargada correctamente.", "Perceptron");
				return true;
			}
			catch (Exception e)
			{
				MessageBox.Show($"No se ha podido cargar la configuración.\nError:\n {e.Message}\nStackTrace:\n{e.StackTrace}");
				return false; 
			}
		}

		/// <summary>
		/// Método para guardar los datos.
		/// </summary>
		public void GuardarDatos()
		{
			try
			{
				for (int i = 0; i < pesos.Length; i++)
				{
					archivo.EscribirArchivo($"Capa {i}:", VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos);
					for (int j = 0; j < pesos[i].Length; j++)
					{
						for (int k = 0; k < pesos[i][j].Length; k++)
						{
							archivo.EscribirArchivo($"Peso[{i}][{j}][{k}] = {pesos[i][j][k]}", VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos);
						}
					}
				}

				for (int i = 0; i < bias.Length; i++)
				{
					archivo.EscribirArchivo($"Capa {i}:", VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos);
					for (int j = 0; j < bias[i].Length; j++)
					{
						archivo.EscribirArchivo($"Sesgo[{i}][{j}] = {bias[i][j]}", VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos);
					}
				}

				MessageBox.Show("Se ha guardado la configuración correctamente.", "Perceptron");
			}
			catch (Exception e)
			{
				MessageBox.Show($"No se ha podido guardar la configuración.\nError:\n {e.Message}");
			}
		}
	}
}