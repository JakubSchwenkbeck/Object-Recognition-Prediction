package AI_Model.Layers;

import java.util.ArrayList;
import java.util.List;

/**
 * Max Pooling Layer implementation for a neural network.
 * This layer performs max pooling operation on the input, which reduces its spatial dimensions.
 */
public class MaxPoolLayer extends Layer {

    private int _stepSize;   // The stride or step size for the pooling operation
    private int _windowSize; // The size of the pooling window (e.g., 2x2)

    private int _inLength;   // Number of input channels (depth)
    private int _inRows;     // Number of rows in the input
    private int _inCols;     // Number of columns in the input

    private List<int[][]> _lastMaxRow; // Stores row indices of max values during the forward pass
    private List<int[][]> _lastMaxCol; // Stores column indices of max values during the forward pass

    /**
     * Constructs a Max Pooling Layer.
     *
     * @param stepSize   The stride or step size for pooling.
     * @param windowSize The size of the pooling window (e.g., 2x2).
     * @param inLength   The number of input channels.
     * @param inRows     The number of rows in the input.
     * @param inCols     The number of columns in the input.
     */
    public MaxPoolLayer(int stepSize, int windowSize, int inLength, int inRows, int inCols) {
        this._stepSize = stepSize;
        this._windowSize = windowSize;
        this._inLength = inLength;
        this._inRows = inRows;
        this._inCols = inCols;
    }

    /**
     * Performs max pooling on the input data.
     *
     * @param input List of input matrices to pool.
     * @return List of pooled matrices.
     */
    public List<double[][]> maxPoolForwardPass(List<double[][]> input) {
        List<double[][]> output = new ArrayList<>();
        _lastMaxRow = new ArrayList<>();
        _lastMaxCol = new ArrayList<>();

        for (double[][] matrix : input) {
            output.add(pool(matrix));
        }

        return output;
    }

    /**
     * Applies max pooling operation on a single matrix.
     *
     * @param input The input matrix to pool.
     * @return The pooled matrix.
     */
    private double[][] pool(double[][] input) {
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

                for (int x = 0; x < _windowSize; x++) {
                    for (int y = 0; y < _windowSize; y++) {
                        int row = r * _stepSize + x;
                        int col = c * _stepSize + y;
                        if (row < _inRows && col < _inCols && input[row][col] > max) {
                            max = input[row][col];
                            maxRows[r][c] = row;
                            maxCols[r][c] = col;
                        }
                    }
                }

                output[r][c] = max;
            }
        }

        _lastMaxRow.add(maxRows);
        _lastMaxCol.add(maxCols);

        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> outputPool = maxPoolForwardPass(input);
        return _nextLayer.getOutput(outputPool);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixList = vectorToMatrix(input, _inLength, _inRows, _inCols);
        return getOutput(matrixList);
    }

    @Override
    public void backPropagation(double[] dLdO) {
        List<double[][]> matrixList = vectorToMatrix(dLdO, getOutputLength(), getOutputRows(), getOutputCols());
        backPropagation(matrixList);
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        List<double[][]> dXdL = new ArrayList<>();
        int l = 0;

        for (double[][] array : dLdO) {
            double[][] error = new double[_inRows][_inCols];

            for (int r = 0; r < getOutputRows(); r++) {
                for (int c = 0; c < getOutputCols(); c++) {
                    int max_i = _lastMaxRow.get(l)[r][c];
                    int max_j = _lastMaxCol.get(l)[r][c];

                    if (max_i != -1) {
                        error[max_i][max_j] += array[r][c];
                    }
                }
            }

            dXdL.add(error);
            l++;
        }

        if (_previousLayer != null) {
            _previousLayer.backPropagation(dXdL);
        }
    }

    @Override
    public int getOutputLength() {
        return _inLength;
    }

    @Override
    public int getOutputRows() {
        return (_inRows - _windowSize) / _stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (_inCols - _windowSize) / _stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return _inLength * getOutputRows() * getOutputCols();
    }
}
