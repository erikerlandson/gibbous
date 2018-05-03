package com.manyangled.gibbous.optim.convex;

import org.apache.commons.math3.optim.OptimizationData;

public class LogBarrierFunction extends ConvexFunction
    implements OptimizationData {

    @Override
    public double value(final double[] x) {
        return 0.0;
    }

    @Override
    void grad(final double[] x, double[] g) {
    }

    @Override
    void hess(final double[] x, double[][] h) {
    }
}
