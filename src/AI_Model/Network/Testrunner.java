package AI_Model.Network;

import AI_Model.Data.PascalVOCDataLoader;
import AI_Model.Data.TrainingSample;
import org.opencv.core.Size;

import java.util.List;

public class Testrunner {

    public static void main() {
        NeuralNetwork net =   NeuralNetwork.buildNetwork(256,256,1,123);
        String xmlDirPath = "C:/Users/jakub/IdeaProjects/PascalVOC_Data/VOC2012_test/Annotations";
        String imageDirPath = "C:/Users/jakub/IdeaProjects/PascalVOC_Data/VOC2012_test/JPEGImages";
        Size targetSize = new Size(256, 256); // Adjust to CNN input size

        List<PascalVOCDataLoader> dataLoaders =PascalVOCDataLoader.loadDir(xmlDirPath,5);
        List<TrainingSample> trainingSamples = PascalVOCDataLoader.loadAndPreprocessImages(dataLoaders, imageDirPath, targetSize,5);
      // System.out.println(net.test(trainingSamples,dataLoaders));
    }
}
