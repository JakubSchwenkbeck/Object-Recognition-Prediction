package AI_Model.Layers;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Abstract base class for neural network layers.
 * Provides methods for handling layer connections, output calculation, and backpropagation.
 */
public abstract class Layer implements Serializable {

    /**
     * The next layer in the network.
     */
    protected Layer nextLayer;

    /**
     * The previous layer in the network.
     */
    protected Layer previousLayer;

    /**
     * Gets the next layer in the network.
     *
     * @return The next layer.
     */
    public Layer getNextLayer() {
        return nextLayer;
    }

    /**
     * Sets the next layer in the network.
     *
     * @param nextLayer The next layer to set.
     */
    public void setNextLayer(Layer nextLayer) {
        this.nextLayer = nextLayer;
    }

    /**
     * Gets the previous layer in the network.
     *
     * @return The previous layer.
     */
    public Layer getPreviousLayer() {
        return previousLayer;
    }

    /**
     * Sets the previous layer in the network.
     *
     * @param previousLayer The previous layer to set.
     */
    public void setPreviousLayer(Layer previousLayer) {
        this.previousLayer = previousLayer;
    }

    /**
     * Calculates the output of the layer given a list of 3D matrices as input.
     *
     * @param input The input data as a list of 3D matrices.
     * @return The output of the layer as a 1D array.
     */
    public abstract double[] computeOutput(List<double[][]> input);

    /**
     * Calculates the output of the layer given a 1D array as input.
     *
     * @param input The input data as a 1D array.
     * @return The output of the layer as a 1D array.
     */
    public abstract double[] computeOutput(double[] input);

    /**
     * Performs backpropagation given the derivative of the loss with respect to the output.
     *
     * @param dLdO The derivative of the loss with respect to the output as a 1D array.
     */
    public abstract void backpropagate(double[] dLdO);

    /**
     * Performs backpropagation given the derivative of the loss with respect to the output
     * for a list of 3D matrices.
     *
     * @param dLdO The derivative of the loss with respect to the output as a list of 3D matrices.
     */
    public abstract void backpropagate(List<double[][]> dLdO);

    /**
     * Gets the length of the output.
     *
     * @return The length of the output.
     */
    public abstract int getOutputLength();

    /**
     * Gets the number of rows in the output.
     *
     * @return The number of rows in the output.
     */
    public abstract int getOutputRows();

    /**
     * Gets the number of columns in the output.
     *
     * @return The number of columns in the output.
     */
    public abstract int getOutputCols();

    /**
     * Gets the total number of elements in the output.
     *
     * @return The total number of elements in the output.
     */
    public abstract int getTotalOutputElements();

    /**
     * Converts a list of 3D matrices to a 1D array.
     *
     * @param input The list of 3D matrices to convert.
     * @return The resulting 1D array.
     */
    public double[] convertMatrixToVector(List<double[][]> input) {
        int length = input.size();
        int rows = input.get(0).length;
        int cols = input.get(0)[0].length;

        double[] vector = new double[length * rows * cols];
        int i = 0;

        for (int l = 0; l < length; l++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    vector[i++] = input.get(l)[r][c];
                }
            }
        }

        return vector;
    }

    /**
     * Converts a 1D array to a list of 3D matrices.
     *
     * @param input  The 1D array to convert.
     * @param length The number of 3D matrices.
     * @param rows   The number of rows in each 3D matrix.
     * @param cols   The number of columns in each 3D matrix.
     * @return The resulting list of 3D matrices.
     */
    public List<double[][]> convertVectorToMatrix(double[] input, int length, int rows, int cols) {
        List<double[][]> output = new ArrayList<>();
        int index = 0;

        for (int l = 0; l < length; l++) {
            double[][] matrix = new double[rows][cols];

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    matrix[r][c] = input[index++];
                }
            }

            output.add(matrix);
        }

        return output;
    }
}
