package com.manyangled.gibbous.optim.convex;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;

public abstract class ConvexFunction implements MultivariateFunction {
    public abstract int dim();

    public abstract double value(final RealVector x);
    public abstract RealVector gradient(final RealVector x);
    public abstract RealMatrix hessian(final RealVector x);
    
    public RealVector gradient(final double[] x) {
        return gradient(new ArrayRealVector(x, false));
    }

    public RealMatrix hessian(final double[] x) {
        return hessian(new ArrayRealVector(x, false));
    }

    @Override
    public double value(final double[] x) {
        return value(new ArrayRealVector(x, false));
    }
}
