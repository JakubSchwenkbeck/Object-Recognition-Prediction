package AI_Model.Layers;

import java.util.List;
import java.util.Random;

/**
 * Fully Connected Layer implementation for a neural network.
 * This layer performs a linear transformation followed by a ReLU activation.
 */
public class FullyConnectedLayer extends Layer {

    private long SEED; // Seed for weight initialization
    private final double leak = 0.01; // Leakiness for Leaky ReLU activation

    private double[][] _weights; // Weight matrix for the layer
    private int _inLength; // Number of input units
    private int _outLength; // Number of output units
    private double _learningRate; // Learning rate for weight updates

    private double[] lastZ; // Last computed linear transformation (z)
    private double[] lastX; // Last input to the layer

    /**
     * Constructs a Fully Connected Layer.
     *
     * @param _inLength    Number of input units
     * @param _outLength   Number of output units
     * @param SEED         Seed for weight initialization
     * @param learningRate Learning rate for weight updates
     */
    public FullyConnectedLayer(int _inLength, int _outLength, long SEED, double learningRate) {
        this._inLength = _inLength;
        this._outLength = _outLength;
        this.SEED = SEED;
        this._learningRate = learningRate;

        _weights = new double[_inLength][_outLength];
        lastZ = new double[_outLength]; // Initialize lastZ
        lastX = new double[_inLength]; // Initialize lastX
        setRandomWeights(); // Initialize weights randomly
    }

    /**
     * Performs a forward pass through the fully connected layer.
     *
     * @param input The input array
     * @return The output array after applying ReLU activation
     */
    public double[] fullyConnectedForwardPass(double[] input) {
        lastX = input;

        double[] z = new double[_outLength];
        double[] out = new double[_outLength];

        // Compute the linear transformation (z = input * weights)
        for (int j = 0; j < _outLength; j++) {
            z[j] = 0; // Initialize z[j]
            for (int i = 0; i < _inLength; i++) {
                z[j] += input[i] * _weights[i][j];
            }
        }

        lastZ = z;

        // Apply ReLU activation
        for (int j = 0; j < _outLength; j++) {
            out[j] = reLu(z[j]);
        }

        return out;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vector = matrixToVector(input);
        return getOutput(vector);
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPass = fullyConnectedForwardPass(input);

        if (_nextLayer != null) {
            return _nextLayer.getOutput(forwardPass);
        } else {
            return forwardPass;
        }
    }

    @Override
    public void backPropagation(double[] dLdO) {
        if (dLdO.length != _outLength) {
            throw new IllegalArgumentException("Size of dLdO must match the output length of the layer.");
        }

        double[] dLdX = new double[_inLength];

        double dOdz; // Derivative of output with respect to z
        double dzdw; // Derivative of z with respect to weights
        double dLdw; // Gradient of loss with respect to weights
        double dzdx; // Derivative of z with respect to input

        // Compute gradients and update weights
        for (int k = 0; k < _inLength; k++) {
            double dLdX_sum = 0;

            for (int j = 0; j < _outLength; j++) {
                dOdz = derivativeReLu(lastZ[j]);
                dzdw = lastX[k];
                dzdx = _weights[k][j];

                dLdw = dLdO[j] * dOdz * dzdw;
                _weights[k][j] -= dLdw * _learningRate;

                dLdX_sum += dLdO[j] * dOdz * dzdx;
            }

            dLdX[k] = dLdX_sum;
        }

        if (_previousLayer != null) {
            _previousLayer.backPropagation(dLdX);
        }
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        double[] vector = matrixToVector(dLdO);
        backPropagation(vector);
    }

    @Override
    public int getOutputLength() {
        return _outLength;
    }

    @Override
    public int getOutputRows() {
        return 1; // Fully connected layers are 1D in output
    }

    @Override
    public int getOutputCols() {
        return _outLength;
    }

    @Override
    public int getOutputElements() {
        return _outLength;
    }

    /**
     * Initializes weights with random values using a Gaussian distribution.
     */
    public void setRandomWeights() {
        Random random = new Random(SEED);
        double stddev = 0.01; // Small standard deviation for weights

        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                _weights[i][j] = stddev * random.nextGaussian();
            }
        }
    }

    /**
     * Applies the ReLU activation function.
     *
     * @param input The input value.
     * @return The activated output value.
     */
    public double reLu(double input) {
        return Math.max(0, input);
    }

    /**
     * Computes the derivative of the ReLU activation function.
     *
     * @param input The input value.
     * @return The derivative of ReLU, which is either `leak` or `1`.
     */
    public double derivativeReLu(double input) {
        return input <= 0 ? leak : 1;
    }
}
