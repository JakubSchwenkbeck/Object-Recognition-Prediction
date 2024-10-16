package AI_Model.Data;

import org.opencv.core.Mat;

public class TrainingSample {
    private final Mat image;
    private final PascalVOCDataLoader.BoundingBox boundingBox;

    private final int Class = 0;
    public TrainingSample(Mat image, PascalVOCDataLoader.BoundingBox boundingBox) {
        this.image = image;
        this.boundingBox = boundingBox;
    }

    public Mat getImage() {
        return image;
    }
    public int ReturnClass(){
    return Class;
    }

    public PascalVOCDataLoader.BoundingBox getBoundingBox() {
        return boundingBox;
    }
}
