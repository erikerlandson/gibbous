package com.manyangled.gibbous.optim.convex;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.OpenMapRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.optim.OptimizationData;

public class LinearFunction extends ConvexFunction
    implements OptimizationData {
    private final RealVector b;
    private final double c;
    private final int n;

    LinearFunction(RealVector b, double c) {
        int d = b.getDimension();
        if (d < 1) throw new IllegalArgumentException("Dimension must be nonzero");
        this.b = b;
        this.c = c;
        this.n = d;
    }

    LinearFunction(double[] b, double c) {
        this(new ArrayRealVector(b), c);
    }

    @Override
    public int dim() { return n; }

    @Override
    public double value(final RealVector x) {
        return c + b.dotProduct(x);
    }

    @Override
    public RealVector gradient(final RealVector x) {
        return b.copy();
    }

    @Override
    public RealMatrix hessian(final RealVector x) {
        // the Hessian is just zero for a linear function
        return new OpenMapRealMatrix(n, n);
    }
}
