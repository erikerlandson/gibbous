package com.manyangled.gibbous.optim.convex;

import java.lang.Math;
import java.util.Collection;
import java.util.Iterator;
import java.util.Arrays;
import org.apache.commons.math3.linear.MatrixUtils;
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
        int d = b.length;
        if (d < 1) throw new IllegalArgumentException("Dimension must be nonzero");
        if (A.length != d) throw new DimensionMismatchException(A.length, d);
        this.A = new Array2DRowRealMatrix(A);
        MatrixUtils.checkSymmetric(this.A, 1e-6);
        this.b = new ArrayRealVector(b);
        this.c = c;
        n = d;
    }

    @Override
    public int dim() { return n; }

    @Override
    public double value(final double[] x) {
        if (x.length != n) throw new DimensionMismatchException(x.length, n);
        RealVector vx = new ArrayRealVector(x);
        RealVector Ax = new ArrayRealVector(A.operate(vx));
        double v = 0.5 * Ax.dotProduct(vx);
        v += b.dotProduct(vx);
        v += c;
        return v;
    }

    @Override
    public void fillGradient(final double[] x, double[] g) {
        if (x.length != n) throw new DimensionMismatchException(x.length, n);
        if (g.length != n) throw new DimensionMismatchException(g.length, n);
        double[] Ax = A.operate(x);
        for (int j = 0; j < n; ++j) {
            g[j] = Ax[j] + b.getEntry(j);
        }
    }

    @Override
    public void fillHessian(final double[] x, double[][] h) {
        if (x.length != n) throw new DimensionMismatchException(x.length, n);
        if (h.length != n) throw new DimensionMismatchException(h.length, n);
        for (int j = 0; j < n;  ++j) {
            if (h[j].length != n) throw new DimensionMismatchException(h[j].length, n);
            for (int k = 0; k < n; ++k) h[j][k] = A.getEntry(j, k);
        }
    }
}
