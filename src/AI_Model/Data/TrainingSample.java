package AI_Model.Data;

import org.opencv.core.Mat;

public class TrainingSample {
    private final Mat image;
    private final PascalVOCDataLoader.BoundingBox boundingBox;

    public TrainingSample(Mat image, PascalVOCDataLoader.BoundingBox boundingBox) {
        this.image = image;
        this.boundingBox = boundingBox;
    }

    public Mat getImage() {
        return image;
    }

    public PascalVOCDataLoader.BoundingBox getBoundingBox() {
        return boundingBox;
    }
}
