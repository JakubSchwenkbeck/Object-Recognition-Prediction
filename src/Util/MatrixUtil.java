package Util;

public class MatrixUtil {

    public static double[][] add(double[][] a, double[][] b){
        // going with the assumption of square shaped, not null matrices
        double[][] res = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
               res[i][j] = a[i][j] + b[i][j];
            }

        }


        return res;
    }

}
