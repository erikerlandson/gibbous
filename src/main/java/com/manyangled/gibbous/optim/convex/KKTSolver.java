package com.manyangled.gibbous.optim.convex;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

// solve block factored matrix equation:
// | H AT | | v | = -| g |
// | A  0 | | w |    | h |
// where (v, w) are primal/dual delta-x and delta-nu from algorithm 10.2
// This is an abstract class that can be overridden to take advantage of
// structure in the hessian matrix H, if desired.
public abstract class KKTSolver {
    public abstract KKTSolution solve(
        final RealMatrix H,
        final RealMatrix A, final RealMatrix AT,
        final RealVector g, final RealVector h);
}
