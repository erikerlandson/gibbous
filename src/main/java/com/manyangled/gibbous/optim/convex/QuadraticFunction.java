package com.manyangled.gibbous.optim.convex;

import java.lang.Math;
import java.util.Collection;
import java.util.Iterator;
import java.util.Arrays;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.exception.DimensionMismatchException;

public class QuadraticFunction extends ConvexFunction
    implements OptimizationData {
    private final RealMatrix A;
    private final RealVector b;
    private final double c;
    private final int n;
    
    QuadraticFunction(double[][] A, double[] b, double c) {
        this.A = new Array2DRowRealMatrix(A);
        this.b = new ArrayRealVector(b);
        this.c = c;
        n = A.length;
    }

    @Override
    public int dim() { return n; }

    @Override
    public double value(final double[] x) { return 0.0; }
    
    @Override
    public void fillGradient(final double[] x, double[] g) {
        Arrays.fill(g, 0.0);
    }

    @Override
    public void fillHessian(final double[] x, double[][] h) {
        for (double[] row: h) Arrays.fill(row, 0.0);
    }
}
