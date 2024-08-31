package AI_Model.Network;

import AI_Model.Data.*;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.CvType;

import java.util.ArrayList;
import java.util.List;


public class TrainingRunner {
    public static void main(String[] args) {
        String imageDir = "path/to/images";
        String annotationDir = "path/to/annotations";

        PascalVOCDataLoader dataLoader = new PascalVOCDataLoader();
        List<Mat> images = dataLoader.loadImages(imageDir);
        List<ObjectAnnotation> annotations = dataLoader.loadAnnotations(annotationDir);

        DataPreprocessor.prepareTrainingData(images, annotations);

        // Initialize and train your neural network
        NeuralNetwork network = NeuralNetwork.buildNetwork(224, 224, 1.0, 1234L); // Example parameters
        network.train(images, annotations);
        
        // Evaluate the network
        float accuracy = network.test(images, annotations);
        System.out.println("Accuracy: " + accuracy);
    }
}
