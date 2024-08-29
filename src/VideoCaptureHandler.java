import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;

/**
 * The {@code VideoCaptureHandler} class is responsible for capturing video frames
 * from a camera.
 */
public class VideoCaptureHandler {
    private VideoCapture capture;

    /**
     * Initializes the video capture from the specified camera index.
     *
     * @param cameraIndex Index of the camera to use (e.g., 0 for default).
     */
    public VideoCaptureHandler(int cameraIndex) {
        capture = new VideoCapture(cameraIndex);
        if (!capture.isOpened()) {
            throw new RuntimeException("Error: Camera not found or cannot be opened.");
        }
    }

    /**
     * Captures a new frame from the camera.
     *
     * @param frame The {@code Mat} object to store the captured frame.
     * @return {@code true} if the frame is successfully captured; {@code false} otherwise.
     */
    public boolean captureFrame(Mat frame) {
        return capture.read(frame);
    }

    /**
     * Releases the video capture resources.
     */
    public void release() {
        if (capture != null && capture.isOpened()) {
            capture.release();
        }
    }
}
