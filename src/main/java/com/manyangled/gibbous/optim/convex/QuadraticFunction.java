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
import org.apache.commons.math3.exception.DimensionMismatchException;

public class QuadraticFunction extends ConvexFunction {
    private final RealMatrix A;
    private final RealVector b;
    private final double c;
    private final int n;

    QuadraticFunction(RealMatrix A, RealVector b, double c) {
        int d = b.getDimension();
        if (d < 1) throw new IllegalArgumentException("Dimension must be nonzero");
        if (A.getRowDimension() != d)
            throw new DimensionMismatchException(A.getRowDimension(), d);
        MatrixUtils.checkSymmetric(A, 1e-6);
        this.A = A.copy();
        this.b = b.copy();
        this.c = c;
        this.n = d;
    }

    QuadraticFunction(double[][] A, double[] b, double c) {
        this(new Array2DRowRealMatrix(A), new ArrayRealVector(b), c);
    }

    @Override
    public int dim() { return n; }

    @Override
    public double value(final RealVector x) {
        double v = 0.5 * (A.operate(x)).dotProduct(x);
        v += b.dotProduct(x);
        v += c;
        return v;
    }

    @Override
    public RealVector gradient(final RealVector x) {
        return (A.operate(x)).add(b);
    }

    @Override
    public RealMatrix hessian(final RealVector x) {
        return A.copy();
    }
}
