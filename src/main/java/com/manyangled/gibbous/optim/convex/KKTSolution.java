package com.manyangled.gibbous.optim.convex;

import org.apache.commons.math3.linear.RealVector;

public class KKTSolution {
    public final RealVector xDelta;
    public final RealVector nuPlus;

    KKTSolution(final RealVector xd, final RealVector nup) {
        this.xDelta = xd;
        this.nuPlus = nup;
    }
}
