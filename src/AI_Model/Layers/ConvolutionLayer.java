package AI_Model.Layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static Util.MatrixUtil.*;

/**
 * Convolutional Layer implementation for a neural network.
 * This layer applies convolution operations to the input data using a set of learnable filters.
 */
public class ConvolutionLayer extends Layer {

    private final int filterSize;   // Size of the convolutional filters (e.g., 3x3)
    private final int stride;       // Stride for the convolution operation
    private final int inputDepth;   // Number of input channels (depth)
    private final int inputHeight;  // Height of the input
    private final int inputWidth;   // Width of the input
    private final double learningRate;  // Learning rate for gradient updates

    private final List<double[][]> filters;  // List of convolutional filters
    private List<double[][]> lastInput;      // Stores the last input during the forward pass

    /**
     * Constructs a Convolutional Layer.
     *
     * @param filterSize   Size of the convolutional filters (e.g., 3x3).
     * @param stride       Stride for the convolution operation.
     * @param inputDepth   Number of input channels.
     * @param inputHeight  Height of the input.
     * @param inputWidth   Width of the input.
     * @param seed         Seed for random filter initialization.
     * @param numFilters   Number of filters in this layer.
     * @param learningRate Learning rate for gradient updates.
     */
    public ConvolutionLayer(int filterSize, int stride, int inputDepth, int inputHeight, int inputWidth, long seed, int numFilters, double learningRate) {
        this.filterSize = filterSize;
        this.stride = stride;
        this.inputDepth = inputDepth;
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.learningRate = learningRate;

        this.filters = generateRandomFilters(numFilters, seed);
    }

    /**
     * Generates random filters based on the specified seed.
     *
     * @param numFilters Number of filters to generate.
     * @param seed       Seed for random number generation.
     * @return A list of randomly initialized filters.
     */
    private List<double[][]> generateRandomFilters(int numFilters, long seed) {
        List<double[][]> filters = new ArrayList<>();
        Random random = new Random(seed);

        for (int n = 0; n < numFilters; n++) {
            double[][] filter = new double[filterSize][filterSize];
            for (int i = 0; i < filterSize; i++) {
                for (int j = 0; j < filterSize; j++) {
                    filter[i][j] = random.nextGaussian();
                }
            }
            filters.add(filter);
        }
        return filters;
    }

    /**
     * Performs the forward pass of the convolutional layer.
     *
     * @param input List of input matrices.
     * @return List of output matrices after applying the convolution operation.
     */
    public List<double[][]> forwardPass(List<double[][]> input) {
        this.lastInput = input;

        List<double[][]> output = new ArrayList<>();
        for (double[][] matrix : input) {
            for (double[][] filter : filters) {
                output.add(applyConvolution(matrix, filter));
            }
        }

        return output;
    }

    /**
     * Applies the convolution operation on a single matrix.
     *
     * @param input  The input matrix.
     * @param filter The filter to apply.
     * @return The output matrix after applying the convolution.
     */
    private double[][] applyConvolution(double[][] input, double[][] filter) {
        int outputRows = (input.length - filter.length) / stride + 1;
        int outputCols = (input[0].length - filter[0].length) / stride + 1;
        double[][] output = new double[outputRows][outputCols];

        for (int r = 0; r <= input.length - filter.length; r += stride) {
            for (int c = 0; c <= input[0].length - filter[0].length; c += stride) {
                double sum = 0.0;
                for (int i = 0; i < filter.length; i++) {
                    for (int j = 0; j < filter[0].length; j++) {
                        sum += filter[i][j] * input[r + i][c + j];
                    }
                }
                output[r / stride][c / stride] = sum;
            }
        }
        return output;
    }

    /**
     * Expands the error matrix by inserting zeros between the elements, according to the stride.
     *
     * @param input The input matrix.
     * @return The expanded matrix.
     */
    private double[][] expandMatrix(double[][] input) {
        if (stride == 1) {
            return input;
        }

        int expandedRows = (input.length - 1) * stride + 1;
        int expandedCols = (input[0].length - 1) * stride + 1;
        double[][] expandedMatrix = new double[expandedRows][expandedCols];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                expandedMatrix[i * stride][j * stride] = input[i][j];
            }
        }

        return expandedMatrix;
    }

    @Override
    public double[] computeOutput(List<double[][]> input) {
        List<double[][]> output = forwardPass(input);
        return nextLayer.computeOutput(output);
    }

    @Override
    public double[] computeOutput(double[] input) {
        List<double[][]> inputMatrices = convertVectorToMatrix(input, inputDepth, inputHeight, inputWidth);
        return computeOutput(inputMatrices);
    }

    @Override
    public void backpropagate(double[] dLdO) {
        List<double[][]> gradientMatrices = convertVectorToMatrix(dLdO, getOutputLength(), getOutputRows(), getOutputCols());
        backpropagate(gradientMatrices);
    }

    @Override
    public void backpropagate(List<double[][]> dLdO) {
        List<double[][]> filterGradients = new ArrayList<>();
        List<double[][]> dLdIPreviousLayer = new ArrayList<>();

        for (int f = 0; f < filters.size(); f++) {
            filterGradients.add(new double[filterSize][filterSize]);
        }

        for (int i = 0; i < lastInput.size(); i++) {
            double[][] inputGradient = new double[inputHeight][inputWidth];

            for (int f = 0; f < filters.size(); f++) {
                double[][] currentFilter = filters.get(f);
                double[][] error = dLdO.get(i * filters.size() + f);

                double[][] expandedError = expandMatrix(error);
                double[][] dLdFilter = applyConvolution(lastInput.get(i), expandedError);

                double[][] delta = mul(dLdFilter, -learningRate);
                filterGradients.set(f, add(filterGradients.get(f), delta));

                double[][] flippedError = flipMatrixHorizontal(flipMatrixVertical(expandedError));
                inputGradient = add(inputGradient, fullConvolve(currentFilter, flippedError));
            }

            dLdIPreviousLayer.add(inputGradient);
        }

        for (int f = 0; f < filters.size(); f++) {
            filters.set(f, add(filterGradients.get(f), filters.get(f)));
        }

        if (previousLayer != null) {
            previousLayer.backpropagate(dLdIPreviousLayer);
        }
    }


    /**
     * Performs a full convolution on the input with the given filter (including padding).
     *
     * @param input  The input matrix.
     * @param filter The filter to apply.
     * @return The output matrix after the full convolution.
     */
    private double[][] fullConvolve(double[][] input, double[][] filter) {
        int outputRows = input.length + filter.length - 1;
        int outputCols = input[0].length + filter[0].length - 1;
        double[][] output = new double[outputRows][outputCols];

        for (int i = -filter.length + 1; i < input.length; i++) {
            for (int j = -filter[0].length + 1; j < input[0].length; j++) {
                double sum = 0.0;
                for (int x = 0; x < filter.length; x++) {
                    for (int y = 0; y < filter[0].length; y++) {
                        int inputRow = i + x;
                        int inputCol = j + y;
                        if (inputRow >= 0 && inputCol >= 0 && inputRow < input.length && inputCol < input[0].length) {
                            sum += filter[x][y] * input[inputRow][inputCol];
                        }
                    }
                }
                output[i + filter.length - 1][j + filter[0].length - 1] = sum;
            }
        }
        return output;
    }

    @Override
    public int getOutputLength() {
        return filters.size() * inputDepth;
    }

    @Override
    public int getOutputRows() {
        return (inputHeight - filterSize) / stride + 1;
    }

    @Override
    public int getOutputCols() {
        return (inputWidth - filterSize) / stride + 1;
    }

    @Override
    public int getTotalOutputElements() {
        return getOutputCols() * getOutputRows() * getOutputLength();

    }


}
