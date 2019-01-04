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

import java.util.ArrayList;

import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.nonlinear.scalar.MultivariateOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.util.Pair;
import org.apache.commons.math3.optim.ConvergenceChecker;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;

/**
 * An abstract class that represents a {@link MultivariateOptimizer} capable of
 * optimizing convex functions, possibly under a set of convex constraints.
 */
public abstract class ConvexOptimizer extends MultivariateOptimizer {
    protected TwiceDifferentiableFunction convexObjective;

    protected ConvexOptimizer() {
        super(null);
    }

    @Override
    protected void parseOptimizationData(OptimizationData... optData) {
        super.parseOptimizationData(optData);
        for (OptimizationData data: optData) {
            if (data instanceof ObjectiveFunction) {
                MultivariateFunction f = ((ObjectiveFunction)data).getObjectiveFunction();
                if (f instanceof TwiceDifferentiableFunction) {
                  convexObjective = (TwiceDifferentiableFunction)f;
                } else {
                    throw new IllegalArgumentException("TwiceDifferentiableFunction objective required");
                }
                continue;
            }
        }
        if (convexObjective == null)
            throw new IllegalStateException("Expected a TwiceDifferentiableFunction argument");
    }

    @Override
    public PointValuePair optimize(OptimizationData... optData) {
        return super.optimize(optData);
    }

    /**
     * A {@link ConvergenceChecker} specialized to be used as a halting condition when solving
     * for feasible points. Allows the optimization to halt when the objective function becomes
     * negative.
     * <p>
     * NOTE: This class is primarily intended for internal use by
     * {@link #feasiblePoint(OptimizationData... optData)}
     * <p>
     * A useful property is that objective functions having no finite minimum can
     * be handled via a {@link HaltingCondition} having a {@link NegChecker} payload. An example
     * of such a problem is solving a feasible point for a single planar constraint.
     */
    public static class NegChecker implements ConvergenceChecker<Pair<RealVector, Double> > {
        @Override
        public boolean converged(
            int iter,
            Pair<RealVector, Double> prv, Pair<RealVector, Double> cur) {
            return cur.getSecond() < 0.0;
        }
    }

    /**
     * Solve a feasible point for a set of constraints, or determine if the constraints are not feasible.
     * <p>
     * Implements feasible point algorithm described
     * <a href="http://erikerlandson.github.io/blog/2018/06/03/solving-feasible-points-with-smooth-max/">here.</a>
     * <p>
     * Supports the following {@link OptimizationData} parameters as arguments
     * <ul>
     *   <li>initial guess {@link InitialGuess} - optional: defaults to zero-vector. </li>
     *   <li>convex inequality constraints: {@link InequalityConstraintSet} - optional </li>
     *   <li>linear inequality constraints: {@link LinearInequalityConstraint} - optional </li>
     *   <li>linear equality constraints: {@link LinearEqualityConstraint} - optional </li>
     *   <li>convergence epsilon: {@link ConvergenceEpsilon} - optional </li>
     *   <li>inner optimizer parameters: {@link InnerOptimizationData} - optional: passed down to {@link NewtonOptimizer} inner calls. </li>
     * </ul>
     * <p>
     * NOTE: There must be at least one inequality constraint provided, via {@link LinearInequalityConstraint},
     * {@link InequalityConstraintSet}, etc.
     * <p>
     * NOTE: all parameters are also passed to {@link NewtonOptimizer}, and so
     * for example setting {@link ConvergenceEpsilon} here will also set it for inner calls to {@link NewtonOptimizer}. However, any
     * settings passed via {@link InnerOptimizationData} are applied last for {@link NewtonOptimizer}, and so will have precedence.
     * <p>
     * @param optData list of {@link OptimizationData} arguments
     * @return a {@link PointValuePair} where the first element is a feasible point (x), or the point "nearest to feasible"
     * in the sense of minimizing the maximum distance to a constraint surface. The second value is the maximum
     * value f[k](x) over all given constraint functions f[k]. If this value is negative, then (x) is feasible. If the value
     * is &gt; 0, then the constraints cannot be satisfied simultaneously.
     */
    public static PointValuePair feasiblePoint(OptimizationData... optData) {
        double epsilon = ConvergenceEpsilon.CONVERGENCE_EPSILON_DEFAULT;
        RealVector initialGuess = null;
        ArrayList<OptimizationData> barrierArgs = new ArrayList<OptimizationData>();
        ArrayList<TwiceDifferentiableFunction> ineqConstraints =
            new ArrayList<TwiceDifferentiableFunction>();
        final TwiceDifferentiableFunction[] fType = new TwiceDifferentiableFunction[0];
        final OptimizationData[] odType = new OptimizationData[0];
        final ArrayList<OptimizationData> solverArgs = new ArrayList<OptimizationData>();
        final ArrayList<OptimizationData> innerArgs = new ArrayList<OptimizationData>();
        for (OptimizationData data: optData) {
            if (canPassFromMain(data)) {
                solverArgs.add(data);
            }
            if (data instanceof InitialGuess) {
                initialGuess = new ArrayRealVector(((InitialGuess)data).getInitialGuess());
                continue;
            }
            if (data instanceof LinearInequalityConstraint) {
                for (TwiceDifferentiableFunction f: ((LinearInequalityConstraint)data).lcf)
                    ineqConstraints.add(f);
                continue;
            }
            if (data instanceof InequalityConstraintSet) {
                ineqConstraints.addAll(((InequalityConstraintSet)data).constraints);
                continue;
            }
            if (data instanceof ConvergenceEpsilon) {
                epsilon = ((ConvergenceEpsilon)data).epsilon;
                continue;
            }
            if (data instanceof InnerOptimizationData) {
                for (OptimizationData d: ((InnerOptimizationData)data).optData.toArray(odType))
                    if (canPassFromInner(d)) innerArgs.add(d);
                continue;
            }
        }
        // Inner arguments apply after "main" arguments to the inner solver
        solverArgs.addAll(innerArgs);
        // check to see if any linear equality constraints are in effect
        boolean hasLinearInequalityConstraint = false;
        for (OptimizationData data: solverArgs.toArray(odType))
            if (data instanceof LinearEqualityConstraint) hasLinearInequalityConstraint = true;
        if (ineqConstraints.size() < 1)
            throw new IllegalStateException("set of inequality constraints was empty");
        final int n = ineqConstraints.get(0).dim();
        if (initialGuess == null) initialGuess = new ArrayRealVector(n, 0.0);
        final TwiceDifferentiableFunction[] fk = ineqConstraints.toArray(fType);
        // These are free parameters, and might be exposed to a user, but I'm not currently
        // convinced there's a lot of value to tweaking them.
        final double minNBallFactor = Math.log(1e-3);
        final double minSigma = 10.0;
        final double sigmaFactor = 1.5;
        final double targetTolerance = 0.01;
        // Initialize our location and get the maximum over the constraint functions
        RealVector x = initialGuess;
        double s = fkMax(x.toArray(), fk);
        // If our point is already feasible we are done, unless we need to satisfy
        // linear equality constraints, in which case just run it through the Newton algorithm to make
        // sure they are satisfied
        if (s < 0.0 && !hasLinearInequalityConstraint) return new PointValuePair(x.toArray(), s);
        double alpha = 1.0;
        while (true) {
            //System.out.format("***s= %f  x= %s\n", s, x);
            // Apply scaling to the n-ball constraint so that it doesn't dominate the location
            // of the optimal point overly much. This substantially increases the convergence rate.
            // If the maximum value of the constraint functions grows very large, then that indicates
            // our current distance to our feasible region is large, so have the scale increase
            // accordingly. An n-ball constraint is basically squared distance from center, so scale
            // grows proportionally to sqrt of (s). Sigma-factor is sort of a magic constant hack, but
            // appears to improve convergence performance.
            double sigma = minSigma;
            if (s > 0.0) sigma = Math.max(sigma, sigmaFactor*Math.sqrt(s));
            // add the n-ball constraint, to guarantee a non-singular hessian
            TwiceDifferentiableFunction nbc = QuadraticFunction.nBallConstraintFunction(x, 1.0, 1.0/sigma);
            ArrayList<TwiceDifferentiableFunction> augConstraints =
                (ArrayList<TwiceDifferentiableFunction>)(ineqConstraints.clone());
            augConstraints.add(nbc);
            // if necessary, tune the smooth-max alpha to guarantee that our n-ball hessian
            // isn't totally washed out.
            double v0 = nbc.value(x);
            if (v0 < (s + minNBallFactor)) alpha = minNBallFactor / (v0 - s);
            // return the point (x) that minimizes the maximum value of fk(x), and/or (x)
            // where fk(x) is negative for all constraints fk.
            // Smooth-max over fk is always >= to true max, so if smooth-max becomes negative
            // we know max is negative (feasible).
            ArrayList<OptimizationData> args = (ArrayList<OptimizationData>)solverArgs.clone();
            args.add(new InitialGuess(x.toArray()));
            args.add(new ObjectiveFunction(new SmoothMaxFunction(alpha, augConstraints.toArray(fType))));
            args.add(new HaltingCondition(new NegChecker()));
            PointValuePair spvp = (new NewtonOptimizer()).optimize(args.toArray(odType));
            RealVector xprv = x;
            // update our solution x, and the true maximum of constraint functions
            x = new ArrayRealVector(spvp.getFirst());
            s = fkMax(spvp.getFirst(), fk);
            // if our latest x satisfies all contstraints, we can stop
            if (s < -targetTolerance) break;
            RealVector xdelta = x.subtract(xprv);
            // if we are no longer moving, our augmented n-ball constraint is no longer
            // influencing the result, and we've identified our mini-max point, whether
            // it is feasible or not.
            if (xdelta.dotProduct(xdelta) < epsilon) break;
            // increase alpha as we converge, so that smooth-max more closely approximates true max
            // see: http://erikerlandson.github.io/blog/2019/01/02/the-smooth-max-minimum-incident-of-december-2018/
            alpha *= 10.0;
        }
        return new PointValuePair(x.toArray(), s);
    }

    private static boolean canPassFromMain(OptimizationData data) {
        if (data instanceof InitialGuess) return false;
        if (data instanceof ObjectiveFunction) return false;
        if (data instanceof HaltingCondition) return false;
        return true;
    }

    private static boolean canPassFromInner(OptimizationData data) {
        return canPassFromMain(data);
    }

    private static double fkMax(double[] x, TwiceDifferentiableFunction[] fk) {
        double s = Double.NEGATIVE_INFINITY;
        for (TwiceDifferentiableFunction f: fk) {
            double y = f.value(x);
            if (s < y) s = y;
        }
        return s;
    }

    /**
     * Determine if a {@link RealMatrix} subclass is a dense format or not
     * @param M The matrix to test
     * @return true if matrix is known to be a dense format, false otherwise.
     */
    public static boolean isDense(RealMatrix M) {
        if (M instanceof Array2DRowRealMatrix) return true;
        if (M instanceof BlockRealMatrix) return true;
        return false;
    }
}
