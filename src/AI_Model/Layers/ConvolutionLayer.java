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
     * Filters are initialized with random values sampled from a Gaussian distribution.
     * These filters are used to scan across the input data during the convolution operation.
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
     * The forward pass involves sliding each filter over the input matrix
     * and computing the dot product between the filter and the corresponding region of the input.
     * This operation is applied to each input channel and filter, producing a feature map.
     *
     * @param input List of input matrices (one per input channel).
     * @return List of output matrices (one per feature map).
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
     * Applies the convolution operation on a single input matrix with a single filter.
     * 
     * This method involves iterating over the input matrix with a sliding window approach,
     * multiplying corresponding elements of the filter and input region, and summing the results.
     * The stride determines the step size of the window.
     *
     * @param input  The input matrix.
     * @param filter The filter to apply.
     * @return The output matrix (feature map) after applying the convolution.
     */
    private double[][] applyConvolution(double[][] input, double[][] filter) {
        int outputRows = (input.length - filter.length) / stride + 1;
        int outputCols = (input[0].length - filter[0].length) / stride + 1;
        double[][] output = new double[outputRows][outputCols];

        // Slide the filter across the input matrix
        for (int r = 0; r <= input.length - filter.length; r += stride) {
            for (int c = 0; c <= input[0].length - filter[0].length; c += stride) {
                double sum = 0.0;
                // Element-wise multiplication and summation
                for (int i = 0; i < filter.length; i++) {
                    for (int j = 0; j < filter[0].length; j++) {
                        sum += filter[i][j] * input[r + i][c + j];
                    }
                }
                output[r / stride][c / stride] = sum; // Assign the sum to the output feature map
            }
        }
        return output;
    }

    /**
     * Expands the error matrix by inserting zeros between the elements, according to the stride.
     * 
     * This is used in the backpropagation process when computing gradients for layers
     * with strides greater than 1, ensuring the gradient matches the original input size.
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

        // Insert zeros between elements
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

    /**
     * Backpropagation through the convolutional layer.
     * 
     * This method computes the gradients of the loss with respect to the filters and the input.
     * The gradient with respect to each filter is computed by convolving the error from the next layer
     * with the input from the forward pass.
     * The gradient with respect to the input is computed by convolving the flipped filter
     * with the expanded error from the next layer.
     *
     * @param dLdO List of matrices representing the gradient of the loss with respect to the output.
     */
    @Override
    public void backpropagate(List<double[][]> dLdO) {
        List<double[][]> filterGradients = new ArrayList<>();
        List<double[][]> dLdIPreviousLayer = new ArrayList<>();

        // Initialize gradient matrices for each filter
        for (int f = 0; f < filters.size(); f++) {
            filterGradients.add(new double[filterSize][filterSize]);
        }

        // Compute gradients for each input and corresponding filter
        for (int i = 0; i < lastInput.size(); i++) {
            double[][] inputGradient = new double[inputHeight][inputWidth];

            for (int f = 0; f < filters.size(); f++) {
                double[][] currentFilter = filters.get(f);
                double[][] error = dLdO.get(i * filters.size() + f);

                // Expand the error matrix to match the stride
                double[][] expandedError = expandMatrix(error);

                // Gradient with respect to the filter (dLdFilter)
                double[][] dLdFilter = applyConvolution(lastInput.get(i), expandedError);

                // Update filter gradients
                double[][] delta = mul(dLdFilter, -learningRate);
                filterGradients.set(f, add(filterGradients.get(f), delta));

                // Compute gradient with respect to the input by convolving the flipped filter with the error
                double[][] flippedError = flipMatrixHorizontal(flipMatrixVertical(expandedError));
                inputGradient = add(inputGradient, fullConvolve(currentFilter, flippedError));
            }

            dLdIPreviousLayer.add(inputGradient); // Accumulate gradients for the previous layer's input
        }

        // Update the filters by adding the computed gradients
        for (int f = 0; f < filters.size(); f++) {
            filters.set(f, add(filterGradients.get(f), filters.get(f)));
        }

        // Propagate the gradients to the previous layer
        if (previousLayer != null) {
            previousLayer.backpropagate(dLdIPreviousLayer);
        }
    }

    /**
     * Performs a full convolution on the input with the given filter (including padding).
     * 
     * This method calculates the full convolution by sliding the filter over every possible position of the input,
     * including those where the filter extends beyond the boundaries of the input, effectively padding the input with zeros.
     *
     * @param input  The input matrix.
     * @param filter The filter to apply.
     * @return The output matrix after the full convolution.
     */
    private double[][] fullConvolve(double[][] input, double[][] filter) {
        int outputRows = input.length + filter.length - 1;
        int outputCols = input[0].length + filter[0].length - 1;
        double[][] output = new double[outputRows][outputCols];

        // Perform convolution with padding
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
