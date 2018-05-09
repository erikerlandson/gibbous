package com.manyangled.gibbous.optim.convex;

import org.apache.commons.math3.optim.OptimizationData;

public class Beta implements OptimizationData {
    public final double beta;
    Beta(double b) {
        if ((b <= 0.0) || (b >= 1.0))
            throw new IllegalArgumentException("beta must be on (0,1)");
        this.beta = b;
    }
}
