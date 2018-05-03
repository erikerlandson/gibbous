package com.manyangled.gibbous.optim.convex;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.exception.DimensionMismatchException;

public abstract class ConvexFunction implements MultivariateFunction {
    public abstract int dim();

    public abstract void fillGradient(final double[] x, double[] g);
    public abstract void fillHessian(final double[] x, double[][] h);

    public double[] gradient(final double[] x) {
        if (x.length != dim()) {
            throw new DimensionMismatchException(x.length, dim());
        }
        double[] g = new double[dim()];
        fillGradient(x, g);
        return g;
    }

    public double[][] hessian(final double[] x) {
        if (x.length != dim()) {
            throw new DimensionMismatchException(x.length, dim());
        }
        double[][] h = new double[dim()][dim()];
        fillHessian(x, h);
        return h;
    }
}
