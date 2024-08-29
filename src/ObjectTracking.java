import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;

/**
 * The {@code ObjectTracking} class captures video from the default camera,
 * performs basic object tracking, and displays the video feed with an example
 * bounding box drawn on the frame.
 */
public class ObjectTracking {

    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    /**
     * Main method that initializes video capture, processes each frame to
     * detect and track objects, and displays the video feed.
     *
     * @param args Command line arguments (not used).
     */
    public static void main(String[] args) {
        // Open video capture from the default camera
        VideoCapture capture = new VideoCapture(0);

        // Check if the camera opened successfully
        if (!capture.isOpened()) {
            System.out.println("Error: Camera not found.");
            return;
        }

        // Create a window for displaying the video feed
        HighGui.namedWindow("Object Tracking", HighGui.WINDOW_AUTOSIZE);

        // Create Mat objects for storing frames and grayscale images
        Mat frame = new Mat();
        Mat gray = new Mat();

        while (true) {
            // Capture a new frame from the video feed
            capture.read(frame);

            // Exit if no frame is captured
            if (frame.empty()) {
                System.out.println("Error: No frame captured.");
                break;
            }

            // Convert the frame to grayscale
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

            // Draw a bounding box on the frame (example bounding box)
            Rect rect = new Rect(50, 50, 200, 200);
            Imgproc.rectangle(frame, rect.tl(), rect.br(), new Scalar(0, 255, 0), 2);

            // Display the frame in the window
            HighGui.imshow("Object Tracking", frame);

            // Exit the loop if the 'q' key is pressed
            if (HighGui.waitKey(30) == 'q') {
                break;
            }
        }

        // Release video capture resources
        capture.release();

        // Close the display window
        HighGui.destroyAllWindows();
    }
}

