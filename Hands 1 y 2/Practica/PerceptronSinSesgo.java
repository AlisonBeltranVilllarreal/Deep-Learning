import java.lang.Math;

// Clase PerceptronSinSesgo
public class PerceptronSinSesgo {
    double[] weights; // Pesos sin sesgo
    double learningRate; // Tasa de aprendizaje

    // Constructor para inicializar pesos y tasa de aprendizaje
    public PerceptronSinSesgo(int inputSize, double learningRate) {
        this.weights = new double[inputSize];
        this.learningRate = learningRate;
        initializeWeights(); // Inicialización de pesos aleatorios
    }

    // Inicialización aleatoria de los pesos
    private void initializeWeights() {
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Math.random(); // Pesos entre 0 y 1
        }
    }

    // Función Sigmoid para calcular la salida
    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Predicción del perceptrón sin sesgo
    public int predict(double[] inputs) {
        double weightedSum = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            weightedSum += inputs[i] * weights[i];
        }
        double output = sigmoid(weightedSum);
        return output >= 0.5 ? 1 : 0; // Redondeo a 0 o 1
    }

    // Entrenamiento del perceptrón sin sesgo
    public void train(double[][] inputs, int[] outputs, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("\nEpoca " + (epoch + 1) + " para AND:");
            for (int i = 0; i < inputs.length; i++) {
                double weightedSum = 0.0;
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
                }
            }
        }
    }
}
