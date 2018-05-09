package com.manyangled.gibbous.optim.convex;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.exception.DimensionMismatchException;

public class EqualityConstraint implements OptimizationData {
    public final RealMatrix A;
    public final RealVector b;

    EqualityConstraint(final RealMatrix A, final RealVector b) {
        int k = A.getRowDimension();
        if (b.getDimension() != k)
            throw new DimensionMismatchException(b.getDimension(), k);
        this.A = A;
        this.b = b;
    }

    EqualityConstraint(final double[][] A, final double[] b) {
        this(new Array2DRowRealMatrix(A), new ArrayRealVector(b));
    }
}
