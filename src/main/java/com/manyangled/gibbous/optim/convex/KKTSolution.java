package com.manyangled.gibbous.optim.convex;

import org.apache.commons.math3.linear.RealVector;

public class KKTSolution {
    public final RealVector xDelta;
    public final RealVector nuPlus;
    public final double lambdaSquared;

    // Used for Newton's method with equality constraints (alg 10.2)
    KKTSolution(final RealVector xd, final RealVector nup) {
        this.xDelta = xd;
        this.nuPlus = nup;
        this.lambdaSquared = 0.0;
    }

    // Used for Newton's method without constraints (alg 9.5)
    KKTSolution(final RealVector xd, final double lsq) {
        this.xDelta = xd;
        this.lambdaSquared = lsq;
        this.nuPlus = null;
    }
}
