package com.manyangled.gibbous.optim.convex;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.optim.OptimizationData;

public abstract class KKTSolver implements OptimizationData {
    // solve block factored matrix equation:
    // | H AT | | v | = -| g |
    // | A  0 | | w |    | h |
    // where (v, w) are primal/dual delta-x and "nu+" from algorithm 10.2
    // This is an abstract class that can be overridden to take advantage of
    // structure in the hessian matrix H, if desired.
    public abstract KKTSolution solve(
        final RealMatrix H,
        final RealMatrix A, final RealMatrix AT,
        final RealVector g, final RealVector h);

    // Solve constraint-free system Hv = -g
    // returns delta-x (aka v) and lambda-squared from Algorithm 9.5
    // delta-nu (aka w, aka the dual) is returned as null
    public abstract KKTSolution solve(final RealMatrix H, final RealVector g);
}
