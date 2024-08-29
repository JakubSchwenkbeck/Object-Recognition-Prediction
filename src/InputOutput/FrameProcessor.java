package InputOutput;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

/**
 * The {@code FrameProcessor} class is responsible for processing video frames
 * and displaying them.
 */
public class FrameProcessor {

    /**
     * Displays the given frame in a window.
     *
     * @param frame The {@code Mat} object to be displayed.
     * @param windowName The name of the window to display the frame in.
     */
    public void displayFrame(Mat frame, String windowName) {
        HighGui.imshow(windowName, frame);
    }

    /**
     * Draws a bounding box on the given frame.
     *
     * @param frame The {@code Mat} object on which to draw the bounding box.
     * @param rect The {@code Rect} object representing the bounding box.
     */
    public void drawBoundingBox(Mat frame, Rect rect) {
        Imgproc.rectangle(frame, rect.tl(), rect.br(), new Scalar(0, 255, 0), 2);
    }
}
