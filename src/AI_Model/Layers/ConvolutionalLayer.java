package AI_Model.Layers;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ConvolutionalLayer {

    private long SEED;

    private List<double[][]> _filters;

    private int _filterSize;

    private int _stepSize;

    private int _inLength;

    private int _inRows;

    private int _inCols;

    private double learningRate;


    private List<double[][]> _lastInput;

    public ConvolutionalLayer(int _filterSize, int _stepsize, int _inLength, int _inRows, int _inCols, long SEED, int numFilters, double learningRate) {
        this._filterSize = _filterSize;
        this._stepSize = _stepsize;
        this._inLength = _inLength;
        this._inRows = _inRows;
        this._inCols = _inCols;
        this.SEED = SEED;
        this.learningRate = learningRate;

         generateRandomFilters(numFilters);
    }

    private void generateRandomFilters(int numFilters){
        List<double[][]> filters = new ArrayList<>();
        Random random = new Random(SEED);

        for(int n = 0; n < numFilters; n++) {
            double[][] newFilter = new double[_filterSize][_filterSize];

            for(int i = 0; i < _filterSize; i++){
                for(int j = 0; j < _filterSize; j++){

                    double value = random.nextGaussian();
                    newFilter[i][j] = value;
                }
            }

            filters.add(newFilter);

        }

        _filters = filters;
    }

    public List<double[][]> convolutionForwardPass(List<double[][]> list){
        _lastInput = list;

        List<double[][]> output = new ArrayList<>();

        for (int m = 0; m < list.size(); m++){
            for(double[][] filter : _filters){
                output.add(convolve(list.get(m), filter, _stepSize));
            }

        }

        return output;

    }




    }
