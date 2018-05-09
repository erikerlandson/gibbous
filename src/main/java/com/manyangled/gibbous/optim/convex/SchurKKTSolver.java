package com.manyangled.gibbous.optim.convex;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.DecompositionSolver;

// algorithm 10.3
// solve block factored matrix equation:
// | H AT | | v | = -| g |
// | A  0 | | w |    | h |
// where (v, w) are primal/dual delta-x and delta-nu from algorithm 10.2
public class SchurKKTSolver {
    public KKTSolution solve(
        final RealMatrix H,
        final RealMatrix A, final RealMatrix AT,
        final RealVector g, final RealVector h) {
        CholeskyDecomposition cdH = new CholeskyDecomposition(H);
        DecompositionSolver dsH = cdH.getSolver();
        RealMatrix m1 = dsH.solve(AT);
        RealVector v1 = dsH.solve(g);
        RealMatrix S = A.multiply(m1); // -S relative to 10.3
        CholeskyDecomposition cdS = new CholeskyDecomposition(S);
        DecompositionSolver dsS = cdS.getSolver();
        RealVector w = dsS.solve(h.subtract(A.operate(v1))); // both sides negative, so w same
        RealVector v = dsH.solve(g.add(AT.operate(w))); // this yields -v
        v.mapMultiplyToSelf(-1.0); // correct -v to +v
        return new KKTSolution(w, v);
    }
}
