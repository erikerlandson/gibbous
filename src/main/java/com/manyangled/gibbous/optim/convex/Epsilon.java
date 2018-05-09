package com.manyangled.gibbous.optim.convex;

import org.apache.commons.math3.optim.OptimizationData;

public class Epsilon implements OptimizationData {
    public final double epsilon;
    Epsilon(double eps) {
        if (eps <= 0.0)
            throw new IllegalArgumentException("epsilon must be > 0");
        this.epsilon = eps;
    }
}
