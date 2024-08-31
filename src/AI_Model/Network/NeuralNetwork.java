package AI_Model.Network;

import AI_Model.Layers.ConvolutionLayer;
import AI_Model.Layers.FullyConnectedLayer;
import AI_Model.Layers.Layer;
import AI_Model.Layers.MaxPoolingLayer;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * NeuralNetwork class that processes input frames to recognize objects, specifically humans.
 */
public class NeuralNetwork {

    private final List<Layer> layers;
    private final double scaleFactor;

    /**
     * Constructor for NeuralNetwork.
     *
     * @param layers       List of layers that make up the neural network.
     * @param scaleFactor  Scaling factor applied during inference.
     */
    public NeuralNetwork(List<Layer> layers, double scaleFactor) {
        this.layers = layers;
        this.scaleFactor = scaleFactor;
        linkLayers();
    }

    /**
     * Links the layers together by setting the previous and next layers.
     */
    private void linkLayers() {
        if (layers.size() <= 1) return;

        for (int i = 0; i < layers.size(); i++) {
            if (i == 0) {
                layers.get(i).setNextLayer(layers.get(i + 1));
            } else if (i == layers.size() - 1) {
                layers.get(i).setPreviousLayer(layers.get(i - 1));
            } else {
                layers.get(i).setPreviousLayer(layers.get(i - 1));
                layers.get(i).setNextLayer(layers.get(i + 1));
            }
        }
    }

    /**
     * Builds a neural network by adding layers.
     *
     * @param inputRows     Number of rows in the input image.
     * @param inputCols     Number of columns in the input image.
     * @param scaleFactor   Scaling factor for input normalization.
     * @param seed          Random seed for initializing weights.
     * @return A constructed NeuralNetwork object.
     */
    public static NeuralNetwork buildNetwork(int inputRows, int inputCols, double scaleFactor, long seed) {
        List<Layer> layers = new ArrayList<>();

        // Add layers in a sensible order
        layers.add(new ConvolutionLayer(3, 1, 1, inputRows, inputCols, seed, 32, 0.01));
        layers.add(new MaxPoolingLayer(2, 2, 32, inputRows - 2, inputCols - 2));
        layers.add(new ConvolutionLayer(3, 1, 32, (inputRows - 2) / 2, (inputCols - 2) / 2, seed, 64, 0.01));
        layers.add(new MaxPoolingLayer(2, 2, 64, (inputRows - 4) / 2, (inputCols - 4) / 2));
        layers.add(new FullyConnectedLayer(256, 64, seed,0.01));
        layers.add(new FullyConnectedLayer(64, 5, seed,0.01)); // Assuming 5 output values (class + bounding box)

        return new NeuralNetwork(layers, scaleFactor);
    }

    /**
     * Processes an input frame and attempts to recognize humans.
     *
     * @param frame The input frame from OpenCV.
     * @return A string label ("Human") and the position of the recognized object.
     */
    public String recognizeObject(Mat frame) {
        Mat resizedFrame = preprocessFrame(frame);
        double[] inputVector = convertMatToInputVector(resizedFrame);
        double[] output = forwardPass(inputVector);
        return interpretOutput(output);
    }

    /**
     * Preprocesses the input frame (resize and normalize).
     *
     * @param frame The input frame.
     * @return The preprocessed frame.
     */
    private Mat preprocessFrame(Mat frame) {
        Size size = new Size(layers.get(0).getOutputCols(), layers.get(0).getOutputRows());
        Mat resizedFrame = new Mat();
        Imgproc.resize(frame, resizedFrame, size);
        resizedFrame.convertTo(resizedFrame, Core.CV_32F, scaleFactor / 255.0);
        return resizedFrame;
    }

    /**
     * Converts the preprocessed frame into a 1D input vector for the neural network.
     *
     * @param frame The preprocessed frame.
     * @return The input vector.
     */
    private double[] convertMatToInputVector(Mat frame) {
        int totalElements = (int) (frame.total() * frame.channels());
        double[] inputVector = new double[totalElements];
        frame.get(0, 0, inputVector);
        return inputVector;
    }

    /**
     * Performs a forward pass through the network.
     *
     * @param inputVector The input vector to the network.
     * @return The output vector from the network.
     */
    private double[] forwardPass(double[] inputVector) {
        double[] output = inputVector;
        for (Layer layer : layers) {
            output = layer.computeOutput(output);
        }
        return output;
    }

    /**
     * Interprets the network's output to determine the label and position of the recognized object.
     *
     * @param output The output vector from the network.
     * @return A label ("Human") if recognized, otherwise "None".
     */
    private String interpretOutput(double[] output) {
        int humanLabelIndex = 0;
        double threshold = 0.5;

        if (output[humanLabelIndex] > threshold) {
            // Assuming output contains bounding box data in subsequent indices (x, y, width, height)
            int x = (int) output[1];
            int y = (int) output[2];
            int width = (int) output[3];
            int height = (int) output[4];

            // Return label and bounding box information (as needed)
            return "Human: Position (" + x + ", " + y + "), Size (" + width + "x" + height + ")";
        }
        return "None";
    }

    /**
     * Trains the neural network using a set of training images.
     *
     * @param images List of images to train on.
     */
    public void train(List<Mat> images, List<int[]> labels) {
        for (int i = 0; i < images.size(); i++) {
            Mat image = images.get(i);
            int[] label = labels.get(i);

            Mat resizedFrame = preprocessFrame(image);
            double[] inputVector = convertMatToInputVector(resizedFrame);
            double[] output = forwardPass(inputVector);

            double[] errors = getErrors(output, label);
            backpropagate(errors);
        }
    }

    /**
     * Tests the neural network using a set of test images.
     *
     * @param images List of test images.
     * @return The accuracy of the neural network.
     */
    public float test(List<Mat> images, List<int[]> labels) {
        int correct = 0;

        for (int i = 0; i < images.size(); i++) {
            Mat image = images.get(i);
            int[] label = labels.get(i);

            String prediction = recognizeObject(image);
            if (prediction.equals("Human")) {
                correct++;
            }
        }

        return (float) correct / images.size();
    }

    /**
     * Computes the error between the network's output and the correct label.
     *
     * @param networkOutput The output produced by the network.
     * @param correctLabel  The correct label.
     * @return An array representing the error.
     */
    private double[] getErrors(double[] networkOutput, int[] correctLabel) {
        double[] errors = new double[networkOutput.length];
        for (int i = 0; i < networkOutput.length; i++) {
            errors[i] = correctLabel[i] - networkOutput[i];
        }
        return errors;
    }

    /**
     * Backpropagates the errors through the network and updates the weights.
     *
     * @param errors The errors to backpropagate.
     */
    private void backpropagate(double[] errors) {
        for (int i = layers.size() - 1; i >= 0; i--) {
            errors = layers.get(i).backpropagate(errors);
        }
    }
}
