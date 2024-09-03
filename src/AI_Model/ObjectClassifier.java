package AI_Model;

import org.opencv.core.Mat;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import Network.*;
/**
 * The {@code AI_Model.ObjectClassifier} class contains methods for classifying objects
 * in video frames using AI.
 */
public class ObjectClassifier {

    /**
     * Classifies objects in the given frame.
     *
     * @param frame The {@code Mat} object representing the video frame.
     * @return A string representing the classification result.
     */


    net = NeuralNetwork.loadNetwork("FILEPATH");

    // net.recognizeObject(frame);


    public String classify(Mat frame) {
        // Placeholder
        return "Object Detected";
    }
}
