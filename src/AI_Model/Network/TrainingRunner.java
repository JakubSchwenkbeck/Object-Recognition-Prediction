package AI_Model.Network;

import AI_Model.Data.*;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.CvType;

import java.util.ArrayList;
import java.util.List;


public class TrainingRunner {



    public static void main(String[] args) {
     NeuralNetwork net =   NeuralNetwork.buildNetwork(256,256,1,123);
     List<PascalVOCDataLoader> XML = PascalVOCDataLoader.loadDir("C:/Users/jakub/IdeaProjects/PascalVOC_Data/VOC2012_test/Annotations");

    }
}
