import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;
import org.opencv.core.CvType;

public class ObjectTracking {

    static {

        try {
            // Attempt to load the OpenCV library
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

            System.out.println("OpenCV library loaded successfully.");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Error: Could not load OpenCV library.");
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static void main(String[] args) {
        // Step 1: Test HighGui with a Static Image
        System.out.println("Testing HighGui with a static image...");
        Mat testImage = Mat.ones(400, 400,CvType.CV_8UC3);
        Imgproc.rectangle(testImage, new Point(50, 50), new Point(350, 350), new Scalar(255, 0, 0), 5);
        HighGui.imshow("Test Image", testImage);
        HighGui.waitKey(0);
        HighGui.destroyAllWindows();

        // Step 2: Initialize Camera
        System.out.println("Initializing camera...");
        VideoCapture capture = new VideoCapture(0); // Try changing index to 1 or 2 if it fails

        if (!capture.isOpened()) {
            System.err.println("Error: Camera not found or cannot be opened.");
            return;
        } else {
            System.out.println("Camera initialized successfully.");
        }

        // Step 3: Test Frame Capture
        System.out.println("Testing frame capture...");
        Mat frame = new Mat();
        Mat gray = new Mat();

        while (true) {
            if (!capture.read(frame)) {
                System.err.println("Error: Could not read frame.");
                break;
            }

            if (frame.empty()) {
                System.err.println("Warning: Empty frame captured.");
                continue;
            }

            System.out.println("Frame captured successfully.");

            // Convert the frame to grayscale
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

            // Draw a sample bounding box
            Rect rect = new Rect(50, 50, 200, 200);
            Imgproc.rectangle(frame, rect.tl(), rect.br(), new Scalar(0, 255, 0), 2);

            // Display the frame in the window
            HighGui.imshow("Object Tracking", frame);

            // Exit the loop if the 'q' key is pressed
            if (HighGui.waitKey(30) == 'q') {
                System.out.println("Exit command received. Closing...");
                break;
            }
        }

        // Release resources
        capture.release();
        HighGui.destroyAllWindows();
    }
}
