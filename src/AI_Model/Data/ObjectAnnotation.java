package AI_Model.Data;

import org.opencv.core.Rect;

public class ObjectAnnotation {
    private final String label;
    private final Rect boundingBox;

    public ObjectAnnotation(String label, Rect boundingBox) {
        this.label = label;
        this.boundingBox = boundingBox;
    }

    public String getLabel() {
        return label;
    }

    public Rect getBoundingBox() {
        return boundingBox;
    }
}
