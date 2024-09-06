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

        weights = new double[inputSize+ 1][outputSize+1 ];
        lastZ = new double[outputSize+ 1];
        lastInput = new double[inputSize+ 1];
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
        //System.out.println("New Forwardpass!");
        double[] z = new double[outputSize];
        double[] output = new double[outputSize];

        // Compute the linear transformation (z = input * weights)
        for (int j = 0; j < outputSize; j++) {
            z[j] = 0; // Initialize z[j]
            for (int i = 0; i < inputSize; i++) {
                //System.out.println("Z lenght: " + z.length);
                //System.out.println("in lenght: " + input.length);
                //System.out.println("wei lenght1: " + weights.length);
                //System.out.println("wei lenght2: " + weights[i].length);
                //System.out.println("i " + i);
                //System.out.println("j " + j);
                try {
                    z[j] += input[i] * weights[i][j];
                }catch(ArrayIndexOutOfBoundsException e){
                    if(z.length < j ){
                        z[j] = 0;
                    }
                }
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
    public double[] computeOutput(List<double[][]> input) {
        double[] vector = convertMatrixToVector(input);
        return computeOutput(vector);
    }

    @Override
    public double[] computeOutput(double[] input) {
        double[] forwardPassOutput = forwardPass(input);

        if (nextLayer != null) {
            return nextLayer.computeOutput(forwardPassOutput);
        } else {
            return forwardPassOutput;
        }
    }

    /**
     * Performs backpropagation to compute gradients and update weights.
     * <p>
     * Backpropagation is the process of adjusting the weights in the neural network to minimize the error.
     * It works by propagating the derivative of the loss function with respect to the output backward through the network.
     * </p>
     *
     * @param dLdO The derivative of the loss function with respect to the output of this layer.
     */
    @Override
    public void backpropagate(double[] dLdO) {
        if (dLdO.length != outputSize) {
            throw new IllegalArgumentException("Size of dLdO must match the output length of the layer.");
        }

        double[] dLdX = new double[inputSize]; // Array to hold the gradient with respect to the input of this layer

        // Loop over each input unit
        for (int i = 0; i < inputSize; i++) {
            double dLdXSum = 0;

            // Loop over each output unit
            for (int j = 0; j < outputSize; j++) {
                // Calculate the derivative of the loss with respect to z[j] (dOdz)
                double dOdz = reluDerivative(lastZ[j]);

                // The derivative of z[j] with respect to weight w[i][j] is the input value lastInput[i]
                double dzdw= 0;
                if(lastInput.length > i) {
                     dzdw = lastInput[i];
                }else{
                     dzdw = 1000;
                }

                // The derivative of z[j] with respect to the input x[i] is the weight w[i][j]
                double dzdx = weights[i][j];

                // Calculate the gradient of the loss with respect to the weight (dLdw)
                double dLdw = dLdO[j] * dOdz * dzdw;

                // Update the weight using the calculated gradient
                weights[i][j] -= dLdw * learningRate;

                // Accumulate the gradient with respect to the input
                dLdXSum += dLdO[j] * dOdz * dzdx;
            }

            // Store the gradient with respect to the input
            dLdX[i] = dLdXSum;
        }

        // If there is a previous layer, propagate the gradient back to it
        if (previousLayer != null) {
            previousLayer.backpropagate(dLdX);
        }
    }

    @Override
    public void backpropagate(List<double[][]> dLdO) {
        double[] vector = convertMatrixToVector(dLdO);
        backpropagate(vector);
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
    public int getTotalOutputElements() {
        return outputSize;
    }

    @Override
    public double[] getOutput(List<double[][]> inList) {
        return new double[0];
    }

    /**
     * Initializes weights with random values using a Gaussian distribution.
     * <p>
     * This method initializes the weight matrix with small random values
     * drawn from a Gaussian distribution with a small standard deviation.
     * </p>
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
     * <p>
     * ReLU (Rectified Linear Unit) is a simple activation function that outputs the input directly if it is positive;
     * otherwise, it outputs zero. It introduces non-linearity into the model, allowing it to learn complex patterns.
     * </p>
     *
     * @param input The input value.
     * @return The activated output value (max(0, input)).
     */
    private double relu(double input) {
        return Math.max(0, input);
    }

    /**
     * Computes the derivative of the ReLU activation function.
     * <p>
     * The derivative of the ReLU function is 1 for positive input values and `leakiness` for negative input values.
     * This is used during backpropagation to compute gradients.
     * </p>
     *
     * @param input The input value.
     * @return The derivative of ReLU, which is either `leakiness` or `1`.
     */
    private double reluDerivative(double input) {
        return input <= 0 ? leakiness : 1;
    }
}
