

// Clase principal para ejecutar los perceptrones
public class Main {
    public static void main(String[] args) {
        // Entrenamiento para compuerta AND (sin sesgo)
        double[][] andInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] andOutputs = {0, 0, 0, 1};
        PerceptronSinSesgo andPerceptron = new PerceptronSinSesgo(2, 0.1);
        andPerceptron.train(andInputs, andOutputs, 100);

        System.out.println("\n=== Resultados finales para AND ===");
        for (int i = 0; i < andPerceptron.weights.length; i++) {
            System.out.printf("Peso %d (w%d): %.4f\n", i + 1, i + 1, andPerceptron.weights[i]);
        }

        // Entrenamiento para compuerta OR (con sesgo)
        double[][] orInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] orOutputs = {0, 1, 1, 1};
        PerceptronConSesgo orPerceptron = new PerceptronConSesgo(2, 0.1);
        orPerceptron.train(orInputs, orOutputs, 100);

        System.out.println("\n=== Resultados finales ===");
        for (int i = 0; i < orPerceptron.weights.length; i++) {
            System.out.printf("Peso %d (w%d): %.4f\n", i + 1, i + 1, orPerceptron.weights[i]);
        }
        System.out.printf("Sesgo (b): %.4f\n", orPerceptron.getBias());
    }
}
