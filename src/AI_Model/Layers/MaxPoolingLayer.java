package AI_Model.Layers;

import java.util.ArrayList;
import java.util.List;

/**
 * Max Pooling Layer implementation for a neural network.
 * This layer performs max pooling operation on the input, which reduces its spatial dimensions
 * by selecting the maximum value within a specified window.
 */
public class MaxPoolingLayer extends Layer {

    private final int stride;   // The stride or step size for the pooling operation
    private final int poolSize; // The size of the pooling window (e.g., 2x2)

    private final int inputDepth; // Number of input channels (depth)
    private final int inputHeight; // Number of rows in the input
    private final int inputWidth;  // Number of columns in the input

    private List<int[][]> maxRowIndices; // Stores row indices of max values during the forward pass
    private List<int[][]> maxColIndices; // Stores column indices of max values during the forward pass

    /**
     * Constructs a Max Pooling Layer.
     *
     * @param stride    The stride or step size for pooling.
     * @param poolSize  The size of the pooling window (e.g., 2x2).
     * @param inputDepth  The number of input channels.
     * @param inputHeight The number of rows in the input.
     * @param inputWidth  The number of columns in the input.
     */
    public MaxPoolingLayer(int stride, int poolSize, int inputDepth, int inputHeight, int inputWidth) {
        this.stride = stride;
        this.poolSize = poolSize;
        this.inputDepth = inputDepth;
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
    }

    /**
     * Performs max pooling on the input data.
     *
     * @param input List of input matrices to pool.
     * @return List of pooled matrices.
     */
    public List<double[][]> forwardPass(List<double[][]> input) {
        List<double[][]> output = new ArrayList<>();
        maxRowIndices = new ArrayList<>();
        maxColIndices = new ArrayList<>();

        for (double[][] matrix : input) {
            output.add(applyPooling(matrix));
        }

        return output;
    }

    /**
     * Applies max pooling operation on a single matrix.
     *
     * @param input The input matrix to pool.
     * @return The pooled matrix.
     */
    private double[][] applyPooling(double[][] input) {
        int outputRows = getOutputRows();
        int outputCols = getOutputCols();

        double[][] output = new double[outputRows][outputCols];
        int[][] maxRows = new int[outputRows][outputCols];
        int[][] maxCols = new int[outputRows][outputCols];

        for (int r = 0; r < outputRows; r++) {
            for (int c = 0; c < outputCols; c++) {
                double max = Double.NEGATIVE_INFINITY;
                maxRows[r][c] = -1;
                maxCols[r][c] = -1;

                for (int x = 0; x < poolSize; x++) {
                    for (int y = 0; y < poolSize; y++) {
                        int row = r * stride + x;
                        int col = c * stride + y;
                        if (row < inputHeight && col < inputWidth && input[row][col] > max) {
                            max = input[row][col];
                            maxRows[r][c] = row;
                            maxCols[r][c] = col;
                        }
                    }
                }

                output[r][c] = max;
            }
        }

        maxRowIndices.add(maxRows);
        maxColIndices.add(maxCols);

        return output;
    }

    @Override
    public double[] computeOutput(List<double[][]> input) {
        List<double[][]> pooledOutput = forwardPass(input);
        return nextLayer.computeOutput(pooledOutput);
    }

    @Override
    public double[] computeOutput(double[] input) {
        List<double[][]> inputMatrices = convertVectorToMatrix(input, inputDepth, inputHeight, inputWidth);
        return computeOutput(inputMatrices);
    }

    /**
     * Performs backpropagation to propagate the gradients through the max pooling layer.
     * <p>
     * During backpropagation, only the maximum values selected during the forward pass
     * contribute to the gradient. The gradient is routed to the input position that had the maximum
     * value in the forward pass, while all other positions receive a gradient of zero.
     * </p>
     *
     * @param dLdO The derivative of the loss function with respect to the output of this layer.
     */
    @Override
    public void backpropagate(double[] dLdO) {
        List<double[][]> dLdOMatrices = convertVectorToMatrix(dLdO, getOutputLength(), getOutputRows(), getOutputCols());
        backpropagate(dLdOMatrices);
    }

    /**
     * Backpropagates the gradient through the max pooling layer.
     *
     * @param dLdO List of matrices representing the gradient of the loss with respect to the output of this layer.
     */
    @Override
    public void backpropagate(List<double[][]> dLdO) {
        List<double[][]> dLdX = new ArrayList<>();
        int layerIndex = 0;

        for (double[][] outputGradient : dLdO) {
            double[][] inputGradient = new double[inputHeight][inputWidth];

            for (int r = 0; r < getOutputRows(); r++) {
                for (int c = 0; c < getOutputCols(); c++) {
                    int maxRow = maxRowIndices.get(layerIndex)[r][c];
                    int maxCol = maxColIndices.get(layerIndex)[r][c];

                    if (maxRow != -1) {
                        inputGradient[maxRow][maxCol] += outputGradient[r][c];
                    }
                }
            }

            dLdX.add(inputGradient);
            layerIndex++;
        }

        if (previousLayer != null) {
            previousLayer.backpropagate(dLdX);
        }
    }

    @Override
    public int getOutputLength() {
        return inputDepth;
    }

    @Override
    public int getOutputRows() {
        return (inputHeight - poolSize) / stride + 1;
    }

    @Override
    public int getOutputCols() {
        return (inputWidth - poolSize) / stride + 1;
    }

    @Override
    public int getTotalOutputElements() {
        return inputDepth * getOutputRows() * getOutputCols();
    }
}
