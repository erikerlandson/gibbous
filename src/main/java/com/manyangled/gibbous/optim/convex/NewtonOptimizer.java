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
            if (nDual >= n)
                throw new IllegalArgumentException("Rank of constraints must be < domain dimension");
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
        final int n = convexObjective.dim();
        if ((eqConstraint == null) || (eqConstraint.b.getDimension() < 1)) {
            // constraints Ax = b are empty
            // Algorithm 9.5: Newton's method (unconstrained)
            RealVector x = xStart;
            double v = convexObjective.value(x);
            while (true) {
                incrementIterationCount();
                RealVector grad = convexObjective.gradient(x);
                RealMatrix hess = convexObjective.hessian(x);
                KKTSolution sol = kktSolver.solve(hess, grad);
                if (sol.lambdaSquared <= (2.0 * epsilon)) break;
                RealVector xDelta = sol.xDelta;
                double t = 1.0;
                double gdd = grad.dotProduct(xDelta);
                while (true) {
                    RealVector tx = x.add(xDelta.mapMultiply(t));
                    double tv = convexObjective.value(tx);
                    if (tv <= v + t*alpha*gdd) {
                        x = tx;
                        v = tv;
                        break;
                    }
                    t = beta * t;
                    if (t * alpha < epsilon) break;
                }
            }
            return new PointValuePair(x.toArray(), v);
        } else {
            // constraints Ax = b are non-empty
            // Algorithm 10.2: Newton's method with equality constraints
            final RealMatrix A = eqConstraint.A;
            final RealVector b = eqConstraint.b;
            final RealMatrix AT = A.transpose();
            final int nDual = b.getDimension();
            RealVector nu = new ArrayRealVector(nDual, 0.0);
            RealVector x = xStart;
            while (true) {
                incrementIterationCount();
                RealVector grad = convexObjective.gradient(x);
                double rNorm = residualNorm(x, nu, grad, A, AT, b);
                if (rNorm <= epsilon) {
                    // For properly small epsilon, I believe that (rNorm <= epsilon)
                    // also implies Ax = b to proper tolerance, since the dual component of
                    // the residual is Ax - b.
                    break;
                }
                RealMatrix hess = convexObjective.hessian(x);
                KKTSolution sol = kktSolver.solve(hess, A, AT, grad, A.operate(x).subtract(b));
                RealVector xDelta = sol.xDelta;
                RealVector nuDelta = sol.nuPlus.subtract(nu);
                double t = 1.0;
                while (true) {
                    RealVector tx = x.add(xDelta.mapMultiply(t));
                    RealVector tnu = nu.add(nuDelta.mapMultiply(t));
                    RealVector tgrad = convexObjective.gradient(tx);
                    double tNorm = residualNorm(tx, tnu, tgrad, A, AT, b);
                    if (tNorm <= (1.0 - alpha*t)*rNorm) {
                        x = tx;
                        nu = tnu;
                        break;
                    }
                    t = beta * t;
                    if (t * alpha < epsilon) break;
                }
            }
            return new PointValuePair(x.toArray(), convexObjective.value(x));
        }
    }

    private double residualNorm(
        RealVector x, RealVector nu, RealVector grad,
        RealMatrix A, RealMatrix AT, RealVector b) {
        RealVector r = A.operate(x).subtract(b);
        RealVector rDual = grad.add(AT.operate(nu));
        double rr = r.dotProduct(r) + rDual.dotProduct(rDual);
        return Math.sqrt(rr);
    }
}
