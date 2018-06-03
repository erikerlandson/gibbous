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
        RealVector initialGuess = null;
        ArrayList<OptimizationData> barrierArgs = new ArrayList<OptimizationData>();
        ArrayList<TwiceDifferentiableFunction> ineqConstraints =
            new ArrayList<TwiceDifferentiableFunction>();
        TwiceDifferentiableFunction[] fType = new TwiceDifferentiableFunction[0];
        for (OptimizationData data: optData) {
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
        }
        if (ineqConstraints.size() < 1)
            throw new IllegalStateException("set of inequality constraints was empty");
        final int n = ineqConstraints.get(0).dim();
        if (initialGuess == null) initialGuess = new ArrayRealVector(n, 0.0);
        final TwiceDifferentiableFunction[] fk = ineqConstraints.toArray(fType);
        RealVector x = initialGuess;
        double s = fkMax(x.toArray(), fk);
        if (s < 0.0) return new PointValuePair(x.toArray(), s);
        double r = 1.0;
        while (true) {
            //System.out.format("***r= %f  s= %f  x= %s\n", r, s, x);
            double sigma = 10.0;
            if (s > 0.0) sigma = Math.max(sigma, 1.5*Math.sqrt(s));
            // add the n-ball constraint, to guarantee a non-singular hessian
            TwiceDifferentiableFunction nbc = QuadraticFunction.nBallConstraintFunction(x, r, 1.0/sigma);
            ArrayList<TwiceDifferentiableFunction> augConstraints =
                (ArrayList<TwiceDifferentiableFunction>)(ineqConstraints.clone());
            augConstraints.add(nbc);
            // return the point (x) that minimizes the maximum value of fk(x), and/or (x)
            // where fk(x) is negative for all constraints fk.
            // Smooth-max over fk is always >= to true max, so if smooth-max becomes negative
            // we know max is negative (feasible).
            double v0 = nbc.value(x);
            double alpha = 1.0;
            if (v0 < (s - 8.0)) alpha = -8 / (v0 - s);
            PointValuePair spvp = (new NewtonOptimizer()).optimize(
                new InitialGuess(x.toArray()),
                new ObjectiveFunction(new SmoothMaxFunction(alpha, augConstraints.toArray(fType))),
                new HaltingCondition(new NegChecker()),
                new SVDSchurKKTSolver()
            );
            RealVector xprv = x;
            // update our solution x, and the true maximum of constraint functions
            x = new ArrayRealVector(spvp.getFirst());
            s = fkMax(spvp.getFirst(), fk);
            // if our latest x satisfies all contstraints, we can stop
            if (s < -0.1) break;
            RealVector xdelta = x.subtract(xprv);
            // if we are no longer moving, our augmented n-ball constraint is no longer
            // influencing the result, and we've identified our mini-max point, whether
            // it is feasible or not.
            if (xdelta.dotProduct(xdelta) < 1e-10) break;
        }
        return new PointValuePair(x.toArray(), s);
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
