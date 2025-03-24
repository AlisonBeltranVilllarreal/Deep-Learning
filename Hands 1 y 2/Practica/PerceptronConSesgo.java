import java.lang.Math;

// Clase PerceptronConSesgo
public class PerceptronConSesgo {
    double[] weights; // Pesos
    double bias; // Sesgo (bias)
    double learningRate; // Tasa de aprendizaje

    // Constructor para inicializar pesos, sesgo y tasa de aprendizaje
    public PerceptronConSesgo(int inputSize, double learningRate) {
        this.weights = new double[inputSize];
        this.bias = Math.random(); // Inicialización aleatoria del sesgo
        this.learningRate = learningRate;
        initializeWeights(); // Inicializar pesos aleatorios
    }

    // Inicialización aleatoria de los pesos
    private void initializeWeights() {
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Math.random();
        }
    }

    // Función Sigmoid para calcular la salida
    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Predicción del perceptrón con sesgo
    public int predict(double[] inputs) {
        double weightedSum = bias;
        for (int i = 0; i < inputs.length; i++) {
            weightedSum += inputs[i] * weights[i];
        }
        double output = sigmoid(weightedSum);
        return output >= 0.5 ? 1 : 0; // Redondeo
    }

    // Entrenamiento del perceptrón con sesgo
    public void train(double[][] inputs, int[] outputs, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("\nEpoca " + (epoch + 1) + " para OR:");
            for (int i = 0; i < inputs.length; i++) {
                double weightedSum = bias;
                for (int j = 0; j < inputs[i].length; j++) {
                    weightedSum += inputs[i][j] * weights[j];
                }
                double output = sigmoid(weightedSum);
                int prediction = output >= 0.5 ? 1 : 0;

                System.out.printf("Entrada: %s | Suma: %.4f | Sigmoid: %.4f | Salida: %d | Esperada: %d\n",
                        java.util.Arrays.toString(inputs[i]), weightedSum, output, prediction, outputs[i]);

                // Actualización si hay error
                if (prediction != outputs[i]) {
                    double error = outputs[i] - prediction;
                    for (int j = 0; j < weights.length; j++) {
                        weights[j] += learningRate * error * inputs[i][j];
                    }
                    bias += learningRate * error;
                }
            }
        }
    }

    // Obtener el sesgo después del entrenamiento
    public double getBias() {
        return bias;
    }
}
