package AI_Model.Network;

import layers.Layer;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

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
    }

    /**
     * Processes an input frame and attempts to recognize humans.
     *
     * @param frame The input frame from OpenCV.
     * @return A string label ("Human") and the position of the recognized object.
     */
    public String recognizeObject(Mat frame) {
        // Preprocess the input frame: resize and normalize
        Mat resizedFrame = preprocessFrame(frame);

        // Convert frame to a format compatible with the neural network (e.g., convert to 1D vector)
        double[] inputVector = convertMatToInputVector(resizedFrame);

        // Forward pass through the network
        double[] output = forwardPass(inputVector);

        // Post-process the output to determine the label and position
        String label = interpretOutput(output);

        return label;
    }

    /**
     * Preprocesses the input frame (resize and normalize).
     *
     * @param frame The input frame.
     * @return The preprocessed frame.
     */
    private Mat preprocessFrame(Mat frame) {
        Size size = new Size(layers.get(0).getOutputRows(), layers.get(0).getOutputCols()); // Resize to match input dimensions
        Mat resizedFrame = new Mat();
        Imgproc.resize(frame, resizedFrame, size);
        resizedFrame.convertTo(resizedFrame, Core.CV_32F, scaleFactor / 255.0); // Normalize
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
        // Simple thresholding logic for demonstration purposes
        int humanLabelIndex = 0; // Assuming the human label is at index 0
        double threshold = 0.5;  // Simple threshold for classification

        if (output[humanLabelIndex] > threshold) {
            // Assume output contains bounding box data in subsequent indices (e.g., x, y, width, height)
            int x = (int) output[1];
            int y = (int) output[2];
            int width = (int) output[3];
            int height = (int) output[4];

            return "Human";
        }

        return "None";
    }

 
}
