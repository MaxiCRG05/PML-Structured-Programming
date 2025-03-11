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

		public readonly Archivos archivo;

		private double[][] neuronas; 
		private double[][][] pesos;
		private double[][] bias;

		public PML(int[] layers)
		{
			capas = layers;
			delta = new double[capas.Length][];
			archivo = new Archivos(VariablesGlobales.Ruta);
			IniciarNeuronas();
			IniciarBias();
			IniciarPesos();
		}

		private void IniciarNeuronas()
		{
			neuronas = new double[capas.Length][];

			for (int i = 0; i < capas.Length; i++)
			{
				neuronas[i] = new double[capas[i]];
			}
		}

		//MÉTODO DE XAVIER (GLOROT)
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

		public void Entrenar(double[][] entradas, double[][] salidas, double tazaAprendizaje, int epocas, int min, int max)
		{
			double mejorError = double.MaxValue;
			int epocasSinMejora = 0;
			const int paciencia = 1000;

			for (int epoca = 0; epoca < epocas; epoca++)
			{
				double errorEpoca = 0;

				for (int i = 0; i < entradas.Length; i++)
				{
					Propagación(entradas[i], min, max);
					Retropropagación(salidas[i], tazaAprendizaje);

					for (int j = 0; j < salidas[i].Length; j++)
					{
						errorEpoca += Math.Pow(salidas[i][j] - neuronas[capas.Length - 1][j], 2); 
					}
				}

				errorEpoca /= (entradas.Length * salidas[0].Length);

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

		public double[] Propagación(double[] entradas, int min, int max)
		{
			entradas = NormalizarValores(entradas, min, max);

			for (int i = 0; i < entradas.Length; i++)
			{
				neuronas[0][i] = entradas[i];
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

		private void Retropropagación(double[] salidaEsperada, double tazaAprendizaje)
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
						pesos[c][j][k] += tazaAprendizaje * delta[c + 1][j] * neuronas[c][k];
					}
					bias[c][j] += tazaAprendizaje * delta[c + 1][j];
				}
			}
		}

		//FUNCIÓN SOFTMAX
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

		public double NormalizarValores(double x, int min, int max)
		{
			return (x - min) / (max - min);
		}

		public double[] NormalizarValores(double[] x, int min, int max)
		{
			double[] resultado = new double[x.Length];

			for (int i = 0; i < x.Length; i++)
			{
				resultado[i] = (x[i] - min) / (max - min);
				if (resultado[i] < 0 || resultado[i] > 1)
				{
					throw new Exception($"Valor normalizado fuera de rango: {resultado[i]}");
				}
			}

			return resultado;
		}

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