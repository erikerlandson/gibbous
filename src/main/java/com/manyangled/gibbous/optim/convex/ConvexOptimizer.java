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

    public static class NegChecker implements ConvergenceChecker<Pair<RealVector, Double> > {
        @Override
        public boolean converged(
            int iter,
            Pair<RealVector, Double> prv, Pair<RealVector, Double> cur) {
            return cur.getSecond() < 0.0;
        }
    }

    public static PointValuePair feasiblePoint(OptimizationData... optData) {
        double epsilon = 1e-10;
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
                RealMatrix A = ((LinearInequalityConstraint)data).A;
                RealVector b = ((LinearInequalityConstraint)data).b;
                for (int j = 0; j < b.getDimension(); ++j)
                    ineqConstraints.add(new LinearFunction(A.getRowVector(j), b.getEntry(j)));
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
            double alpha = 1.0;
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

    // known dense Matrix classes
    public static boolean isDense(RealMatrix M) {
        if (M instanceof Array2DRowRealMatrix) return true;
        if (M instanceof BlockRealMatrix) return true;
        return false;
    }
}
