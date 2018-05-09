package com.manyangled.gibbous.optim.convex;

import org.apache.commons.math3.linear.RealVector;

public class KKTSolution {
    public final RealVector xDelta;
    public final RealVector nuDelta;

    KKTSolution(final RealVector xd, final RealVector nud) {
        this.xDelta = xd;
        this.nuDelta = nud;
    }
}
