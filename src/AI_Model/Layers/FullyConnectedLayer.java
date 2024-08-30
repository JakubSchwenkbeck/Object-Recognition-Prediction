package AI_Model.Layers;

import java.util.List;
import java.util.Random;

/**
 * Fully Connected Layer implementation for a neural network.
 * This layer performs a linear transformation followed by a ReLU activation.
 */
public class FullyConnectedLayer extends Layer {

    private final long seed; // Seed for weight initialization
    private final double leakiness = 0.01; // Leakiness for Leaky ReLU activation
    private final double learningRate; // Learning rate for weight updates

    private double[][] weights; // Weight matrix for the layer
    private final int inputSize; // Number of input units
    private final int outputSize; // Number of output units

    private double[] lastZ; // Last computed linear transformation (z)
    private double[] lastInput; // Last input to the layer

    /**
     * Constructs a Fully Connected Layer.
     *
     * @param inputSize    Number of input units
     * @param outputSize   Number of output units
     * @param seed         Seed for weight initialization
     * @param learningRate Learning rate for weight updates
     */
    public FullyConnectedLayer(int inputSize, int outputSize, long seed, double learningRate) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.seed = seed;
        this.learningRate = learningRate;

        weights = new double[inputSize][outputSize];
        lastZ = new double[outputSize];
        lastInput = new double[inputSize];
        initializeWeights(); // Initialize weights randomly
    }

    /**
     * Performs a forward pass through the fully connected layer.
     *
     * @param input The input array
     * @return The output array after applying ReLU activation
     */
    public double[] forwardPass(double[] input) {
        lastInput = input;

        double[] z = new double[outputSize];
        double[] output = new double[outputSize];

        // Compute the linear transformation (z = input * weights)
        for (int j = 0; j < outputSize; j++) {
            z[j] = 0; // Initialize z[j]
            for (int i = 0; i < inputSize; i++) {
                z[j] += input[i] * weights[i][j];
            }
        }

        lastZ = z;

        // Apply ReLU activation
        for (int j = 0; j < outputSize; j++) {
            output[j] = relu(z[j]);
        }

        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vector = convertMatrixToVector(input);
        return getOutput(vector);
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPassOutput = forwardPass(input);

        if (nextLayer != null) {
            return nextLayer.getOutput(forwardPassOutput);
        } else {
            return forwardPassOutput;
        }
    }

    @Override
    public void backPropagation(double[] dLdO) {
        if (dLdO.length != outputSize) {
            throw new IllegalArgumentException("Size of dLdO must match the output length of the layer.");
        }

        double[] dLdX = new double[inputSize];

        // Compute gradients and update weights
        for (int i = 0; i < inputSize; i++) {
            double dLdXSum = 0;

            for (int j = 0; j < outputSize; j++) {
                double dOdz = reluDerivative(lastZ[j]);
                double dzdw = lastInput[i];
                double dzdx = weights[i][j];

                double dLdw = dLdO[j] * dOdz * dzdw;
                weights[i][j] -= dLdw * learningRate;

                dLdXSum += dLdO[j] * dOdz * dzdx;
            }

            dLdX[i] = dLdXSum;
        }

        if (previousLayer != null) {
            previousLayer.backPropagation(dLdX);
        }
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        double[] vector = convertMatrixToVector(dLdO);
        backPropagation(vector);
    }

    @Override
    public int getOutputLength() {
        return outputSize;
    }

    @Override
    public int getOutputRows() {
        return 1; // Fully connected layers produce 1D output
    }

    @Override
    public int getOutputCols() {
        return outputSize;
    }

    @Override
    public int getOutputElements() {
        return outputSize;
    }

    /**
     * Initializes weights with random values using a Gaussian distribution.
     */
    private void initializeWeights() {
        Random random = new Random(seed);
        double stddev = 0.01; // Small standard deviation for weights

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] = stddev * random.nextGaussian();
            }
        }
    }

    /**
     * Applies the ReLU activation function.
     *
     * @param input The input value.
     * @return The activated output value.
     */
    private double relu(double input) {
        return Math.max(0, input);
    }

    /**
     * Computes the derivative of the ReLU activation function.
     *
     * @param input The input value.
     * @return The derivative of ReLU, which is either `leakiness` or `1`.
     */
    private double reluDerivative(double input) {
        return input <= 0 ? leakiness : 1;
    }
}
