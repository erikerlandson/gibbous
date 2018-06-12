/*
Copyright 2018 Erik Erlandson
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package com.manyangled.gibbous.optim.convex;

import org.apache.commons.math3.util.Pair;
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.ConvergenceException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;

/**
 * Implements convex optimization, using Newton's method.
 * Supports linear equality constraints and infeasible starting point<p>
 * (Algorithm 10.2) from Convex Optimization, Boyd and Vandenberghe, Cambridge University Press, 2008.
 * <p>
 * {@link BarrierOptimizer} supports the following {@link OptimizationData} parameters as arguments
 * to {@link #optimize(OptimizationData...)}:
 * <ul>
 *   <li>convex objective function: ObjectiveFunction - mandatory: must contain a {@link TwiceDifferentiableFunction} </li>
 *   <li>initial guess: InitialGuess - mandatory: need not satisfy equality constraints </li>
 *   <li>linear equality constraints: {@link LinearEqualityConstraint} - optional </li>
 *   <li>convergence epsilon: {@link ConvergenceEpsilon} - optional </li>
 *   <li>backtracking alpha: {@link BacktrackAlpha} - optional </li>
 *   <li>backtracking beta: {@link BacktrackBeta} - optional </li>
 *   <li>KKT equations solver: {@link KKTSolver} - optional </li>
 * </ul>
 */
public class NewtonOptimizer extends ConvexOptimizer {
    private LinearEqualityConstraint eqConstraint;
    private KKTSolver kktSolver = new CholeskySchurKKTSolver();
    private RealVector xStart;
    private double epsilon = ConvergenceEpsilon.CONVERGENCE_EPSILON_DEFAULT;
    private double alpha = BacktrackAlpha.BACKTRACK_ALPHA_DEFAULT;
    private double beta = BacktrackBeta.BACKTRACK_BETA_DEFAULT;
    private HaltingCondition halting;

    public NewtonOptimizer() {
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
            if (data instanceof LinearEqualityConstraint) {
                eqConstraint = (LinearEqualityConstraint)data;
                continue;
            }
            if (data instanceof KKTSolver) {
                kktSolver = (KKTSolver)data;
                continue;
            }
            if (data instanceof ConvergenceEpsilon) {
                epsilon = ((ConvergenceEpsilon)data).epsilon;
                continue;
            }
            if (data instanceof BacktrackAlpha) {
                alpha = ((BacktrackAlpha)data).alpha;
                continue;
            }
            if (data instanceof BacktrackBeta) {
                beta = ((BacktrackBeta)data).beta;
                continue;
            }
            if (data instanceof HaltingCondition) {
                halting = (HaltingCondition)data;
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
                //System.out.format("  v=%f  x=%s  g=%s  h= %s  xdel=%s\n", v, x, grad, hess, sol.xDelta);
                if (sol.lambdaSquared <= (2.0 * epsilon)) break;
                RealVector xDelta = sol.xDelta;
                // If the step direction becomes very small that indicates minimum
                if (xDelta.getNorm() < epsilon) break;
                double gdd = grad.dotProduct(xDelta);
                RealVector tx = null;
                double tv = 0.0;
                boolean foundStep = false;
                for (double t = 1.0 ; t >= epsilon ; t *= beta) {
                    tx = x.add(xDelta.mapMultiply(t));
                    tv = convexObjective.value(tx);
                    if (tv == Double.POSITIVE_INFINITY) continue;
                    if (tv <= v + t*alpha*gdd) {
                        foundStep = true;
                        break;
                    }
                }
                // If there was no forward step to make, that indicates minimum
                if (!foundStep) break;
                // Update x,v for next iteration
                RealVector xprv = x;
                double vprv = v;
                x = tx;
                v = tv;
                // Check halting condition if configured
                if ((halting != null) && halting.checker.converged(
                        getIterations(),
                        new Pair<RealVector, Double>(xprv, vprv),
                        new Pair<RealVector, Double>(x, v))) {
                    break;
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
            RealVector x = xStart;
            RealVector nu = new ArrayRealVector(nDual, 0.0);
            double v = convexObjective.value(x);
            while (true) {
                incrementIterationCount();
                RealVector grad = convexObjective.gradient(x);
                double rNorm = residualNorm(x, nu, grad, A, AT, b);
                if (rNorm <= epsilon) break;
                RealMatrix hess = convexObjective.hessian(x);
                KKTSolution sol = kktSolver.solve(hess, A, AT, grad, A.operate(x).subtract(b));
                RealVector xDelta = sol.xDelta;
                RealVector nuDelta = sol.nuPlus.subtract(nu);
                // If step direction delta becomes sufficiently small it indicates minimum
                if ((xDelta.getNorm() + nuDelta.getNorm()) < epsilon) break;
                RealVector tx = null;
                RealVector tnu = null;
                double tv = 0.0;
                boolean foundStep = false;
                for (double t = 1.0 ; t >= epsilon ; t *= beta) {
                    tx = x.add(xDelta.mapMultiply(t));
                    tv = convexObjective.value(tx);
                    if (tv == Double.POSITIVE_INFINITY) continue;
                    tnu = nu.add(nuDelta.mapMultiply(t));
                    RealVector tgrad = convexObjective.gradient(tx);
                    double tNorm = residualNorm(tx, tnu, tgrad, A, AT, b);
                    if (tNorm <= (1.0 - alpha*t)*rNorm) {
                        foundStep = true;
                        break;
                    }
                }
                // If there was no forward step to make, that indicates minimum
                if (!foundStep) break;
                // update for next iteration
                RealVector xprv = x;
                double vprv = v;
                x = tx;
                nu = tnu;
                v = tv;
                // check halting condition, if it was configured
                if ((halting != null) && halting.checker.converged(
                        getIterations(),
                        new Pair<RealVector, Double>(xprv, vprv),
                        new Pair<RealVector, Double>(x, v))) {
                    break;
                }
            }
            return new PointValuePair(x.toArray(), v);
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
