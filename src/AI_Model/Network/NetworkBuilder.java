package AI_Model.Network;

import layers.ConvolutionLayer;
import layers.FullyConnectedLayer;
import layers.Layer;
import layers.MaxPoolLayer;
import layers.AveragePoolLayer;  // Assuming you have this class implemented

import java.util.ArrayList;
import java.util.List;

/**
 * Builder class for constructing a neural network.
 * This class facilitates the step-by-step addition of layers to the network.
 */
public class NeuralNetworkBuilder {

    private NeuralNetwork net;
    private int inputRows;
    private int inputCols;
    private double scaleFactor;
    private List<Layer> layers;

    /**
     * Constructor for NeuralNetworkBuilder.
     *
     * @param inputRows   Number of rows in the input.
     * @param inputCols   Number of columns in the input.
     * @param scaleFactor Scaling factor applied during training.
     */
    public NeuralNetworkBuilder(int inputRows, int inputCols, double scaleFactor) {
        this.inputRows = inputRows;
        this.inputCols = inputCols;
        this.scaleFactor = scaleFactor;
        this.layers = new ArrayList<>();
    }

    /**
     * Adds a convolutional layer to the network.
     *
     * @param numFilters   Number of filters in the convolutional layer.
     * @param filterSize   Size of each filter (e.g., 3x3).
     * @param stepSize     Step size (stride) for the convolution.
     * @param learningRate Learning rate for the layer.
     * @param seed         Seed for random filter initialization.
     */
    public void addConvolutionLayer(int numFilters, int filterSize, int stepSize, double learningRate, long seed) {
        if (layers.isEmpty()) {
            layers.add(new ConvolutionLayer(filterSize, stepSize, 1, inputRows, inputCols, seed, numFilters, learningRate));
        } else {
            Layer prev = layers.get(layers.size() - 1);
            layers.add(new ConvolutionLayer(filterSize, stepSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols(), seed, numFilters, learningRate));
        }
    }

    /**
     * Adds a max pooling layer to the network.
     *
     * @param windowSize Size of the pooling window.
     * @param stepSize   Step size (stride) for pooling.
     */
    public void addMaxPoolLayer(int windowSize, int stepSize) {
        if (layers.isEmpty()) {
            layers.add(new MaxPoolLayer(stepSize, windowSize, 1, inputRows, inputCols));
        } else {
            Layer prev = layers.get(layers.size() - 1);
            layers.add(new MaxPoolLayer(stepSize, windowSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols()));
        }
    }

    /**
     * Adds an average pooling layer to the network.
     *
     * @param windowSize Size of the pooling window.
     * @param stepSize   Step size (stride) for pooling.
     */
    public void addAveragePoolLayer(int windowSize, int stepSize) {
        if (layers.isEmpty()) {
            layers.add(new AveragePoolLayer(stepSize, windowSize, 1, inputRows, inputCols));
        } else {
            Layer prev = layers.get(layers.size() - 1);
            layers.add(new AveragePoolLayer(stepSize, windowSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols()));
        }
    }

    /**
     * Adds a fully connected layer to the network.
     *
     * @param outLength   Number of neurons in the fully connected layer.
     * @param learningRate Learning rate for the layer.
     * @param seed        Seed for random weight initialization.
     */
    public void addFullyConnectedLayer(int outLength, double learningRate, long seed) {
        if (layers.isEmpty()) {
            layers.add(new FullyConnectedLayer(inputCols * inputRows, outLength, seed, learningRate));
        } else {
            Layer prev = layers.get(layers.size() - 1);
            layers.add(new FullyConnectedLayer(prev.getTotalOutputElements(), outLength, seed, learningRate));
        }
    }

    /**
     * Builds and returns the neural network with the specified layers.
     *
     * @return The constructed neural network.
     */
    public NeuralNetwork build() {
        net = new NeuralNetwork(layers, scaleFactor);
        return net;
    }
}
