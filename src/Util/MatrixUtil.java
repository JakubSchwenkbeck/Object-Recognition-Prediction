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

    public static double[] add(double[] a, double[] b){
        // going with the assumption of square shaped, not null matrices
        double[] res = new double[a.length];

        for (int i = 0; i < a.length; i++) {

                res[i] = a[i] + b[i];
            }




        return res;
    }

    public static double[][] mul(double[][] a, double scalar){
        // going with the assumption of square shaped, not null matrices
        double[][] res = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                res[i][j] = a[i][j] * scalar;
            }

        }


        return res;
    }
    public static double[] mul(double[] a, double scalar){
        // going with the assumption of square shaped, not null matrices
        double[] res = new double[a.length];

        for (int i = 0; i < a.length; i++) {

            res[i] = a[i] *scalar;
        }


        return res;
    }

    /**
     * Flips a matrix horizontally (around its vertical axis).
     *
     * @param matrix The matrix to flip.
     * @return The horizontally flipped matrix.
     */
    public static double[][] flipMatrixHorizontal(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] flipped = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flipped[rows - i - 1][j] = matrix[i][j];
            }
        }
        return flipped;
    }

    /**
     * Flips a matrix vertically (around its horizontal axis).
     *
     * @param matrix The matrix to flip.
     * @return The vertically flipped matrix.
     */
    public static double[][] flipMatrixVertical(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] flipped = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flipped[i][cols - j - 1] = matrix[i][j];
            }
        }
        return flipped;
    }

}
