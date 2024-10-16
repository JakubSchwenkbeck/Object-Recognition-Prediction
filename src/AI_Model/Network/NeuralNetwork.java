package AI_Model.Network;

import AI_Model.Layers.*;
import AI_Model.Data.*;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.CvType;

import java.io.*;
import java.util.*;

import static Util.MatrixUtil.*;
import static org.opencv.core.Core.multiply;

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
    }

    public NeuralNetwork loadNetwork(String filePath) {
    NeuralNetwork network = null;
    try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
        network = (NeuralNetwork) ois.readObject();
        System.out.println("Neural network loaded from " + filePath);
    } catch (IOException | ClassNotFoundException e) {
        e.printStackTrace();
    }
    return network;
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
    private static void linkLayers(List<Layer> layers) {
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
       /* layers.add(new ConvolutionLayer(3, 1, 1, inputRows, inputCols, seed, 32, 0.01)); // 1st Conv Layer
        layers.add(new MaxPoolingLayer(2, 2, 32, inputRows - 2, inputCols - 2)); // Max Pooling Layer 1

        layers.add(new ConvolutionLayer(3, 1, 32, (inputRows - 2) / 2, (inputCols - 2) / 2, seed, 64, 0.01)); // 2nd Conv Layer
       // layers.add(new AveragePoolingLayer(2, 2, 64, (inputRows - 4) / 2, (inputCols - 4) / 2)); // Avg Pooling Layer 2
        layers.add(new MaxPoolingLayer(2, 2, 64, inputRows - 2, inputCols - 2)); // Max Pooling Layer 2

        layers.add(new ConvolutionLayer(3, 1, 64, (inputRows - 6) / 4, (inputCols - 6) / 4, seed, 128, 0.01)); // 3rd Conv Layer
        layers.add(new MaxPoolingLayer(2, 2, 128, (inputRows - 8) / 4, (inputCols - 8) / 4)); // Max Pooling Layer 3

        layers.add(new FullyConnectedLayer(512, 128, seed, 0.01)); // 1st Fully Connected Layer
        layers.add(new FullyConnectedLayer(128, 5, seed, 0.01)); // 2nd Fully Connected Layer (output: label + bounding box)
*/
        // Initial Convolutional Layer: 32 filters, 3x3 kernel, stride of 1, no padding
    /*    layers.add(new ConvolutionLayer(3, 1, 1, inputRows, inputCols, seed, 32, 0.01));

// Max Pooling Layer: 2x2 pooling, 32 input channels, reduces the feature map size by half
        layers.add(new MaxPoolingLayer(2, 2, 32, inputRows - 2, inputCols - 2));

// Second Convolutional Layer: 64 filters, 3x3 kernel, stride of 1, no padding
        layers.add(new ConvolutionLayer(3, 1, 32, (inputRows - 2) / 2, (inputCols - 2) / 2, seed, 64, 0.01));

// Max Pooling Layer: 2x2 pooling, 64 input channels, reduces the feature map size by half again
        layers.add(new MaxPoolingLayer(2, 2, 64, (inputRows - 4) / 2, (inputCols - 4) / 2));

// Fully Connected Layer: 128 neurons connected to the flattened output of the previous layer
        layers.add(new FullyConnectedLayer(128, 64, seed, 0.01));

// Output Layer: 5 outputs (for label and bounding box coordinates)
        layers.add(new FullyConnectedLayer(64, 5, seed, 0.01));
*/
        // First Convolutional Layer
        layers.add(new ConvolutionLayer(3, 3, 1, 256, 256, seed, 32, 0.01));

// First Max Pooling Layer
        layers.add(new MaxPoolingLayer(2, 2, 32, 254, 254));

// Second Convolutional Layer
        layers.add(new ConvolutionLayer(3, 1, 32, 127, 127, seed, 64, 0.01));

// Second Max Pooling Layer
        layers.add(new MaxPoolingLayer(2, 2, 64, 125, 125));

// First Fully Connected Layer
        layers.add(new FullyConnectedLayer(246016, 128, seed, 0.01));

// Output Layer
        layers.add(new FullyConnectedLayer(128, 25, seed, 0.01));
        NeuralNetwork.linkLayers(layers);
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
        // Convert the frame to a 64-bit floating point matrix if it's not already
        Mat convertedFrame = new Mat();
        frame.convertTo(convertedFrame, CvType.CV_64F);

        // Flatten the matrix into a one-dimensional array
        int totalElements = (int) (convertedFrame.total() * convertedFrame.channels());
        double[] inputVector = new double[totalElements];

        // Get the data from the matrix
        convertedFrame.get(0, 0, inputVector);

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
        int numClasses = LABEL_TO_INDEX.size(); // Total number of classes
        double threshold = 0.5;

        // Find the class with the highest probability
        int bestClassIndex = -1;
        double highestProbability = 0.0;

        for (int i = 0; i < numClasses; i++) {
            if (output[i] > highestProbability) {
                highestProbability = output[i];
                bestClassIndex = i;
            }
        }

        // Check if the highest probability exceeds the threshold
        if (bestClassIndex >= 0 && highestProbability > threshold) {
            // Extract bounding box coordinates
            int x = (int) output[numClasses];
            int y = (int) output[numClasses + 1];
            int width = (int) output[numClasses + 2];
            int height = (int) output[numClasses + 3];

            // Get the label for the best class index
            String detectedLabel = getKeyFromValue(LABEL_TO_INDEX, bestClassIndex);

            // Return label and bounding box information
            return detectedLabel + ": Position (" + x + ", " + y + "), Size (" + width + "x" + height + ")";
        }

        return "None";
    }

    // Helper method to get the key from the value in the LABEL_TO_INDEX map
    private String getKeyFromValue(Map<String, Integer> map, int value) {
        for (Map.Entry<String, Integer> entry : map.entrySet()) {
            if (entry.getValue().equals(value)) {
                return entry.getKey();
            }
        }
        return "Unknown";
    }




    /*public void train(List<TrainingSample> trainingSamples) {
        for (TrainingSample sample : trainingSamples) {
            Mat image = sample.getImage();
            System.out.println("Loaded Image!");

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
            System.out.println("Trained!!");
        }

    }

    /**
     * Tests the neural network using a set of test images and annotations.
     *
     * @param images    List of test images.
     * @param annotations List of PascalVOCAnnotation objects corresponding to the images.
     * @return The accuracy of the neural network.
     */
    /*public float test(List<TrainingSample> testsamples,List<PascalVOCDataLoader> annotations) {
        int correct = 0;

        for (int i = 0; i < annotations.size(); i++) {
            Mat image = testsamples.get(i).getImage();
            PascalVOCDataLoader annotation = annotations.get(i);

            String prediction = recognizeObject(image);
            if (prediction.equals("person")) {
                correct++;
            }
        }

        return (float) correct / annotations.size();
    }
/*
    /**
     * Computes the error between the network's output and the correct label.
     *
     * @param networkOutput The output produced by the network.
     * @param correctLabel  The correct label.
     * @return An array representing the error.

    private double[] getErrors(double[] networkOutput, double[] correctLabel) {
        double[] errors = new double[networkOutput.length];
        for (int i = 0; i < networkOutput.length; i++) {
            errors[i] = correctLabel[i] - networkOutput[i];
        }
        return errors;
    }
*/



    public double[] getErrors(double[] networkOutput, int correctClass, double[] trueBoundingBox) {
        // The networkOutput array now contains both classification and bounding box coordinates.
        // Assuming the first 20 outputs are the class scores, and the remaining 4 are bounding box coordinates.
        int numClasses = 20;  // Adjust for number of object classes
        int totalOutput = numClasses + 4;  // Class scores + 4 bounding box values

        double[] errors = new double[totalOutput];

        // Classification error (cross-entropy loss or a simpler approximation)
        double[] expectedClass = new double[numClasses];
        expectedClass[correctClass] = 1;
        double[] classificationError = add(Arrays.copyOfRange(networkOutput, 0, numClasses), mul(expectedClass, -1));

        // Bounding box error (mean squared error for each coordinate: x, y, width, height)
        double[] predictedBoundingBox = Arrays.copyOfRange(networkOutput, numClasses, totalOutput);
        double[] boundingBoxError = add(predictedBoundingBox, mul(trueBoundingBox, -1));  // assuming we want simple difference

        // Combine the classification and bounding box errors
        System.arraycopy(classificationError, 0, errors, 0, numClasses);
        System.arraycopy(boundingBoxError, 0, errors, numClasses, 4);

        return errors;
    }

    private int getMaxIndex(double[] in) {
        double max = 0;
        int index = 0;
        for (int i = 0; i < in.length; i++) {
            if (in[i] >= max) {
                max = in[i];
                index = i;
            }
        }
        return index;
    }

    public int guessClass(double[] networkOutput) {
        // The first 20 outputs represent the classification.
        double[] classScores = Arrays.copyOfRange(networkOutput, 0, 20);
        return getMaxIndex(classScores);  // Return the class with the highest score
    }

    public double[] guessBoundingBox(double[] networkOutput) {
        // The last 4 outputs represent the bounding box.
        return Arrays.copyOfRange(networkOutput, 20, 24);  // Return the bounding box coordinates
    }

    public float test(List<TrainingSample> trainingSamples) {
        int correct = 0;
        for (TrainingSample TS : trainingSamples) {
            List<Mat> inList = new ArrayList<>();
            inList.add(TS.getImage());

            double[] out = layers.get(0).getOutput(convertMatToInputVector(TS.getImage()));
            int guessedClass = guessClass(out);

            // Optional: Test bounding box prediction accuracy here if needed
            // double[] guessedBoundingBox = guessBoundingBox(out);

            if (guessedClass == TS.ReturnClass()) { // annot
                correct++;
            }
        }
        return ((float) correct / trainingSamples.size());
    }

    public double[][] matToDoubleArray(Mat mat) {
        // Get the number of rows and columns from the Mat
        int rows = mat.rows();
        int cols = mat.cols();

        // Initialize a double[][] array to store the values
        double[][] array = new double[rows][cols];

        // Check if the Mat type is CV_64F (for double precision floating point)
        //if (mat.type() == CvType.CV_64F) {
            // Buffer to store the data from the Mat
            double[] data = new double[cols];

            // Iterate over each row and copy the data to the array
            for (int i = 0; i < rows; i++) {
                // Get the ith row data and store in the buffer
                mat.get(i, 0, data);
                // Copy the buffer into the respective row of the array
                System.arraycopy(data, 0, array[i], 0, cols);
            }
        else {
            throw new IllegalArgumentException("Input Mat type must be CV_64F (double precision).");
        }

        return array;
    }


    public void train(List<TrainingSample> trainingSamples) {
        for (TrainingSample TS : trainingSamples) {
            List<Mat> inList = new ArrayList<>();
            inList.add(TS.getImage());

            double[] out = layers.get(0).getOutput(inList);

            // Prepare the correct class label and true bounding box coordinates
            int correctClass = TS.ReturnClass();
            PascalVOCDataLoader.BoundingBox trueBoundingBox = TS.getBoundingBox();  // Assuming this method exists in Image class

            // Calculate the error
            double[] dldO = getErrors(out, correctClass, trueBoundingBox);

            // Backpropagate the errors
            layers.get(layers.size() - 1).backpropagate(dldO);
        }
    }

}
