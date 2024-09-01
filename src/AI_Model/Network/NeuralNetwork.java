package AI_Model.Network;

import AI_Model.Layers.*;
import AI_Model.Data.*;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.CvType;

import java.util.HashMap;
import java.util.Map;

import static Util.MatrixUtil.*;

import java.io.Serializable;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Class representing a neural network.
 */
public class NeuralNetwork implements Serializable {
    private static final long serialVersionUID = 1L;

    private final List<Layer> layers;
    private final double scaleFactor;

    private static final Map<String, Integer> LABEL_TO_INDEX = new HashMap<>();

    static {
        // Initialize your class labels here
        LABEL_TO_INDEX.put("person", 0);
        LABEL_TO_INDEX.put("bird", 1);
        LABEL_TO_INDEX.put("cat", 2);
        LABEL_TO_INDEX.put("cow", 3);
        LABEL_TO_INDEX.put("dog", 4);
        LABEL_TO_INDEX.put("horse", 5);
        LABEL_TO_INDEX.put("sheep", 6);
        LABEL_TO_INDEX.put("aeroplane", 7);
        LABEL_TO_INDEX.put("bicycle", 8);
        LABEL_TO_INDEX.put("boat", 9);
        LABEL_TO_INDEX.put("bus", 10);
        LABEL_TO_INDEX.put("car", 11);
        LABEL_TO_INDEX.put("motorbike", 12);
        LABEL_TO_INDEX.put("train", 13);
        LABEL_TO_INDEX.put("bottle", 14);
        LABEL_TO_INDEX.put("chair", 15);
        LABEL_TO_INDEX.put("dining table", 16);
        LABEL_TO_INDEX.put("potted plant", 17);
        LABEL_TO_INDEX.put("sofa", 18);
        LABEL_TO_INDEX.put("tvmonitor", 19);
    }

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

    public void saveNetwork(NeuralNetwork network, String filePath) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(network);
            System.out.println("Neural network saved to " + filePath);
        } catch (IOException e) {
            e.printStackTrace();
        }
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

        // Add layers in a sensible order with alternating pooling types
        layers.add(new ConvolutionLayer(3, 1, 1, inputRows, inputCols, seed, 32, 0.01)); // 1st Conv Layer
        layers.add(new MaxPoolingLayer(2, 2, 32, inputRows - 2, inputCols - 2)); // Max Pooling Layer 1

        layers.add(new ConvolutionLayer(3, 1, 32, (inputRows - 2) / 2, (inputCols - 2) / 2, seed, 64, 0.01)); // 2nd Conv Layer
        layers.add(new AveragePoolingLayer(2, 2, 64, (inputRows - 4) / 2, (inputCols - 4) / 2)); // Avg Pooling Layer 2

        layers.add(new ConvolutionLayer(3, 1, 64, (inputRows - 6) / 4, (inputCols - 6) / 4, seed, 128, 0.01)); // 3rd Conv Layer
        layers.add(new MaxPoolingLayer(2, 2, 128, (inputRows - 8) / 4, (inputCols - 8) / 4)); // Max Pooling Layer 3

        layers.add(new FullyConnectedLayer(512, 128, seed, 0.01)); // 1st Fully Connected Layer
        layers.add(new FullyConnectedLayer(128, 5, seed, 0.01)); // 2nd Fully Connected Layer (output: label + bounding box)

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
        resizedFrame.convertTo(resizedFrame, CvType.CV_8UC3, scaleFactor / 255.0);
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
     * Trains the neural network using a set of training images and annotations.
     *
     * @param images    List of images to train on.
     * @param annotations List of PascalVOCAnnotation objects corresponding to the images.
     */

    public void train(List<TrainingSample> trainingSamples) {
        for (TrainingSample sample : trainingSamples) {
            Mat image = sample.getImage();
            PascalVOCDataLoader.BoundingBox boundingBox = sample.getBoundingBox();

            Mat resizedFrame = preprocessFrame(image);
            double[] inputVector = convertMatToInputVector(resizedFrame);

            // Prepare the output vector
            int numClasses = LABEL_TO_INDEX.size(); // Number of classes is now 20
            double[] outputVector = new double[numClasses + 4]; // numClasses for one-hot encoding + 4 for bbox

            // Set the one-hot encoded label
            Integer classIndex = LABEL_TO_INDEX.get(boundingBox.getLabel());
            if (classIndex != null) {
                outputVector[classIndex] = 1.0; // Set the corresponding index to 1.0
            }

            // Set bounding box coordinates
            outputVector[numClasses] = boundingBox.getXmin();
            outputVector[numClasses + 1] = boundingBox.getYmin();
            outputVector[numClasses + 2] = boundingBox.getXmax() - boundingBox.getXmin();
            outputVector[numClasses + 3] = boundingBox.getYmax() - boundingBox.getYmin();

            // Perform forward pass and update weights (backpropagation)
            double[] networkOutput = forwardPass(inputVector);
            double[] dldO = getErrors(networkOutput, outputVector);
            layers.get(layers.size() - 1).backpropagate(dldO);
        }

    }

    /**
     * Tests the neural network using a set of test images and annotations.
     *
     * @param images    List of test images.
     * @param annotations List of PascalVOCAnnotation objects corresponding to the images.
     * @return The accuracy of the neural network.
     */
    public float test(List<Mat> images, List<PascalVOCDataLoader> annotations) {
        int correct = 0;

        for (int i = 0; i < images.size(); i++) {
            Mat image = images.get(i);
            PascalVOCDataLoader annotation = annotations.get(i);

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
    private double[] getErrors(double[] networkOutput, double[] correctLabel) {
        double[] errors = new double[networkOutput.length];
        for (int i = 0; i < networkOutput.length; i++) {
            errors[i] = correctLabel[i] - networkOutput[i];
        }
        return errors;
    }


}
