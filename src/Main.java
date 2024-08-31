import AI_Model.Data.PascalVOCDataLoader;
import AI_Model.ObjectClassifier;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.CvType;
import org.opencv.core.Scalar;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;
import org.opencv.core.Core;
import InputOutput.*;
import AI_Model.*;

import java.io.FileNotFoundException;

/**
 * The {@code ObjectTrackingApp} class initializes the components and coordinates
 * video capture, frame processing, and object classification.
 */
public class Main {

    public static void main(String[] args) {
        // Initialize OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
try {
    System.out.println(PascalVOCDataLoader.parseXML("\"C:\\Users\\jakub\\OneDrive\\Desktop\\annot.xml\"").getImageFileName());
}catch(Exception e){
    System.out.println("Parsing not working");
}

        // Create instances of handler classes
        VideoCaptureHandler captureHandler = new VideoCaptureHandler(0);
        FrameProcessor frameProcessor = new FrameProcessor();
        ObjectClassifier classifier = new ObjectClassifier();

        // Create a Mat object for capturing frames
        Mat frame = new Mat();
        Rect sampleRect = new Rect(50, 50, 200, 200); // Example bounding box

        // Test HighGui with a static image
        Mat testImage = Mat.ones(400, 400, CvType.CV_8UC3);
        Imgproc.rectangle(testImage, new Point(50, 50), new Point(350, 350), new Scalar(255, 0, 0), 5);
        HighGui.imshow("Test Image", testImage);
        HighGui.waitKey(0);
        HighGui.destroyAllWindows();

        // Initialize Camera and start processing
        System.out.println("Starting video capture...");
        while (true) {
            if (!captureHandler.captureFrame(frame)) {
                System.err.println("Error: Could not read frame.");
                break;
            }

            if (frame.empty()) {
                System.err.println("Warning: Empty frame captured.");
                continue;
            }

            // Process frame
            frameProcessor.drawBoundingBox(frame, sampleRect);

            // Classify objects
            String result = classifier.classify(frame);
            System.out.println("Classification Result: " + result);

            // Display the frame
            frameProcessor.displayFrame(frame, "Object Tracking");

            // Exit if 'q' is pressed
            if (HighGui.waitKey(30) == 'q') {
                System.out.println("Exit command received. Closing...");
                break;
            }
        }

        // Release resources
        captureHandler.release();
        HighGui.destroyAllWindows();
    }
}
