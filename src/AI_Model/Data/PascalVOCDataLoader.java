package AI_Model.Data;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.File;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.List;

/**
 * Class to process Pascal VOC XML annotation files.
 */
public class PascalVOCDataLoader{

    private final String imageFileName;
    private final List<BoundingBox> boundingBoxes;

    /**
     * Constructor for PascalVOCAnnotation.
     *
     * @param imageFileName The filename of the image.
     * @param boundingBoxes List of bounding boxes for objects in the image.
     */
    public PascalVOCDataLoader(String imageFileName, List<BoundingBox> boundingBoxes) {
        this.imageFileName = imageFileName;
        this.boundingBoxes = boundingBoxes;
    }

    public String getImageFileName() {
        return imageFileName;
    }

    public List<BoundingBox> getBoundingBoxes() {
        return boundingBoxes;
    }

    /**
     * Loads and preprocesses images from a list of PascalVOCDataLoader objects.
     *
     * @param dataLoaders A list of PascalVOCDataLoader objects.
     * @param imageDirPath The directory where images are located.
     * @param targetSize The target size for image resizing.
     * @return A list of TrainingSample objects.
     */
    public static List<TrainingSample> loadAndPreprocessImages(List<PascalVOCDataLoader> dataLoaders, String imageDirPath, Size targetSize, int maxImages) {
        List<TrainingSample> trainingSamples = new ArrayList<>();
        int counter = 0;

        for (PascalVOCDataLoader dataLoader : dataLoaders) {
            if (counter >= maxImages) {
                break; // Stop processing if the maximum number of images is reached
            }

            System.out.println("Loading Dataset Nr : " + counter);
            String imageFilePath = imageDirPath + File.separator + dataLoader.getImageFileName();
            Mat image = Imgcodecs.imread(imageFilePath);

            if (image.empty()) {
                System.err.println("Failed to load image: " + imageFilePath);
                continue;
            }

            // Preprocess the image
            Mat preprocessedImage = preprocessImage(image, targetSize);

            for (PascalVOCDataLoader.BoundingBox bbox : dataLoader.getBoundingBoxes()) {
                TrainingSample sample = new TrainingSample(preprocessedImage, bbox);
                trainingSamples.add(sample);
            }

            counter++; // Increment the counter for each successfully processed image
        }

        return trainingSamples;
    }

    /**
     * Preprocesses an image by resizing and normalizing.
     *
     * @param image The original image.
     * @param targetSize The target size for resizing.
     * @return The preprocessed image.
     */
    public static Mat preprocessImage(Mat image, Size targetSize) {
        Mat resizedImage = new Mat();
        Imgproc.resize(image, resizedImage, targetSize);

        // Normalize the image (optional)
        resizedImage.convertTo(resizedImage, CvType.CV_32F, 1.0 / 255.0);

        return resizedImage;
    }


    /**
     * Parses an XML annotation file and extracts the image filename and bounding boxes.
     *
     * @param xmlFilePath The path to the XML annotation file.
     * @return A PascalVOCAnnotation object containing the extracted information.
     */
    public static PascalVOCDataLoader parseXML(String xmlFilePath) {
        try {
            File xmlFile = new File(xmlFilePath);
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            DocumentBuilder builder = factory.newDocumentBuilder();
            Document document = builder.parse(xmlFile);

            document.getDocumentElement().normalize();

            String imageFileName = "";
            List<BoundingBox> boundingBoxes = new ArrayList<>();

            NodeList sizeList = document.getElementsByTagName("size");
            if (sizeList.getLength() > 0) {
                Node sizeNode = sizeList.item(0);
                if (sizeNode.getNodeType() == Node.ELEMENT_NODE) {
                    Element sizeElement = (Element) sizeNode;
                    // You can extract width, height, depth if needed
                }
            }

            NodeList objectList = document.getElementsByTagName("object");
            for (int i = 0; i < objectList.getLength(); i++) {
                Node objectNode = objectList.item(i);
                if (objectNode.getNodeType() == Node.ELEMENT_NODE) {
                    Element objectElement = (Element) objectNode;

                    // Extract object name
                    String name = objectElement.getElementsByTagName("name").item(0).getTextContent();

                    // Extract bounding box coordinates
                    Element bndboxElement = (Element) objectElement.getElementsByTagName("bndbox").item(0);
                    int xmin = Integer.parseInt(bndboxElement.getElementsByTagName("xmin").item(0).getTextContent());
                    int ymin = Integer.parseInt(bndboxElement.getElementsByTagName("ymin").item(0).getTextContent());
                    int xmax = Integer.parseInt(bndboxElement.getElementsByTagName("xmax").item(0).getTextContent());
                    int ymax = Integer.parseInt(bndboxElement.getElementsByTagName("ymax").item(0).getTextContent());

                    BoundingBox boundingBox = new BoundingBox(name, xmin, ymin, xmax, ymax);
                    boundingBoxes.add(boundingBox);
                }
            }

            // Extract image filename
            NodeList filenameList = document.getElementsByTagName("filename");
            if (filenameList.getLength() > 0) {
                Node filenameNode = filenameList.item(0);
                if (filenameNode.getNodeType() == Node.ELEMENT_NODE) {
                    Element filenameElement = (Element) filenameNode;
                    imageFileName = filenameElement.getTextContent();
                }
            }

            return new PascalVOCDataLoader(imageFileName, boundingBoxes);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
    /**
     * Loads all Pascal VOC XML annotation files from a directory.
     *
     * @param dirPath The path to the directory containing XML files.
     * @return A list of PascalVOCDataLoader objects, each representing an annotation file.
     */
    public static List<PascalVOCDataLoader> loadDir(String dirPath, int maxFiles) {
        List<PascalVOCDataLoader> dataLoaders = new ArrayList<>();
        File dir = new File(dirPath);

        if (dir.isDirectory()) {
            // Filter XML files
            File[] xmlFiles = dir.listFiles(new FilenameFilter() {
                @Override
                public boolean accept(File dir, String name) {
                    return name.endsWith(".xml");
                }
            });

            if (xmlFiles != null) {
                int count = 0;
                for (File xmlFile : xmlFiles) {
                    if (count >= maxFiles) {
                        break; // Stop loading if the maximum number of files is reached
                    }
                    PascalVOCDataLoader dataLoader = parseXML(xmlFile.getAbsolutePath());
                    if (dataLoader != null) {
                        dataLoaders.add(dataLoader);
                        count++;
                    }
                }
            }
        } else {
            System.err.println("Provided path is not a directory: " + dirPath);
        }

        return dataLoaders;
    }

    /**
     * Inner class to represent a bounding box.
     */
    public static class BoundingBox {
        private final String label;
        private final int xmin;
        private final int ymin;
        private final int xmax;
        private final int ymax;

        public BoundingBox(String label, int xmin, int ymin, int xmax, int ymax) {
            this.label = label;
            this.xmin = xmin;
            this.ymin = ymin;
            this.xmax = xmax;
            this.ymax = ymax;
        }

        public String getLabel() {
            return label;
        }

        public int getXmin() {
            return xmin;
        }

        public int getYmin() {
            return ymin;
        }

        public int getXmax() {
            return xmax;
        }

        public int getYmax() {
            return ymax;
        }

        @Override
        public String toString() {
            return "BoundingBox{" +
                    "label='" + label + '\'' +
                    ", xmin=" + xmin +
                    ", ymin=" + ymin +
                    ", xmax=" + xmax +
                    ", ymax=" + ymax +
                    '}';
        }
    }
}
