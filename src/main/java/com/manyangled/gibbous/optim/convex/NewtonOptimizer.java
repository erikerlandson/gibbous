package com.manyangled.gibbous.optim.convex;

import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;

// Algorithm 10.2
public class NewtonOptimizer extends ConvexOptimizer {
    private EqualityConstraint eqConstraint;
    private KKTSolver kktSolver = new SchurKKTSolver();
    private RealVector xStart;
    private double epsilon = 1e-10;
    private double alpha = 0.25;
    private double beta = 0.5;

    protected NewtonOptimizer() {
        super();
    }

    @Override
    public PointValuePair optimize(OptimizationData... optData) {
        return super.optimize(optData);
    }

    @Override
    protected void parseOptimizationData(OptimizationData... optData) {
        super.parseOptimizationData(optData);
        for (OptimizationData data: optData) {
            if (data instanceof EqualityConstraint) {
                eqConstraint = (EqualityConstraint)data;
                continue;
            }
            if (data instanceof KKTSolver) {
                kktSolver = (KKTSolver)data;
                continue;
            }
            if (data instanceof Epsilon) {
                epsilon = ((Epsilon)data).epsilon;
                continue;
            }
            if (data instanceof Alpha) {
                alpha = ((Alpha)data).alpha;
                continue;
            }
            if (data instanceof Beta) {
                beta = ((Beta)data).beta;
                continue;
            }
        }
        // if we got here, convexObjective exists
        int n = convexObjective.dim();
        if (eqConstraint != null) {
            int nDual = eqConstraint.b.getDimension();
            int nTest = eqConstraint.A.getColumnDimension();
            if ((nDual > 0) && (nTest != n))
                throw new DimensionMismatchException(nTest, n);
        }
        if (this.getStartPoint() != null) {
            xStart = new ArrayRealVector(this.getStartPoint());
            if (xStart.getDimension() != n)
                throw new DimensionMismatchException(xStart.getDimension(), n);
        } else {
            xStart = new ArrayRealVector(n, 0.0);
        }
    }

    @Override
    public PointValuePair doOptimize() {
        int n = convexObjective.dim();
        if ((eqConstraint == null) || (eqConstraint.b.getDimension() < 1)) {
            // constraints Ax = b are empty
            return null;
        } else {
            // constraints Ax = b are non-empty
            RealMatrix A = eqConstraint.A;
            RealVector b = eqConstraint.b;
            int nDual = b.getDimension();
            RealVector nuStart = new ArrayRealVector(nDual, 0.0);
            return null;
        }
    }
}
