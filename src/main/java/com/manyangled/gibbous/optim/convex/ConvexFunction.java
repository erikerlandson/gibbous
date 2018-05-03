package com.manyangled.gibbous.optim.convex;

import org.apache.commons.math3.analysis.MultivariateFunction;

public abstract class ConvexFunction implements MultivariateFunction {
    abstract void grad(double[] x, double[] g);
    abstract void hess(double[] x, double[][] h);

    double[] grad(double[] x) {
        double[] g = new double[x.length];
        grad(x, g);
        return g;
    }

    double[][] hess(double[] x) {
        double[][] h = new double[x.length][x.length];
        hess(x, h);
        return h;
    }
}
