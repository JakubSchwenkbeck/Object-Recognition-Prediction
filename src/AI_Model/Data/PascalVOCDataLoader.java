package AI_Model.Data;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.core.CvType;
import org.opencv.core.Size;
import org.xml.sax.InputSource;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.NodeList;
import org.w3c.dom.Node;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class PascalVOCDataLoader {

    public List<Mat> loadImages(String directoryPath) {
        File dir = new File(directoryPath);
        File[] files = dir.listFiles((d, name) -> name.endsWith(".jpg"));
        List<Mat> images = new ArrayList<>();
        for (File file : files) {
            Mat image = Imgcodecs.imread(file.getAbsolutePath());
            images.add(image);
        }
        return images;
    }

    public List<ObjectAnnotation> loadAnnotations(String directoryPath) {
        File dir = new File(directoryPath);
        File[] files = dir.listFiles((d, name) -> name.endsWith(".xml"));
        List<ObjectAnnotation> annotations = new ArrayList<>();
        for (File file : files) {
            try {
                DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
                DocumentBuilder builder = factory.newDocumentBuilder();
                Document doc = builder.parse(new InputSource(file.getAbsolutePath()));
                NodeList objects = doc.getElementsByTagName("object");
                for (int i = 0; i < objects.getLength(); i++) {
                    Node objectNode = objects.item(i);
                    String label = objectNode.getChildNodes().item(1).getTextContent(); // Assuming label is in the second child node
                    NodeList bndbox = objectNode.getChildNodes().item(4).getChildNodes(); // Bounding box info
                    int x = Integer.parseInt(bndbox.item(1).getTextContent());
                    int y = Integer.parseInt(bndbox.item(3).getTextContent());
                    int width = Integer.parseInt(bndbox.item(5).getTextContent()) - x;
                    int height = Integer.parseInt(bndbox.item(7).getTextContent()) - y;
                    Rect boundingBox = new Rect(x, y, width, height);
                    annotations.add(new ObjectAnnotation(label, boundingBox));
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return annotations;
    }
}
