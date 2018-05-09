package com.manyangled.gibbous.optim.convex;

import org.apache.commons.math3.optim.OptimizationData;

public class Alpha implements OptimizationData {
    public final double alpha;
    Alpha(double a) {
        if ((a <= 0.0) || (a >= 0.5))
            throw new IllegalArgumentException("alpha must be on (0,1/2)");
        this.alpha = a;
    }
}
