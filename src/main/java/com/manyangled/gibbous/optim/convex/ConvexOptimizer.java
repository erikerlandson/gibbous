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
        public final int j;
        public NegChecker(int j) {
            this.j = j;
        }
        @Override
        public boolean converged(
            int iter,
            Pair<RealVector, Double> prv, Pair<RealVector, Double> cur) {
            return cur.getFirst().getEntry(j) < 0.0;
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
        }
        if (ineqConstraints.size() < 1)
            throw new IllegalStateException("set of inequality constraints was empty");
        final int n = ineqConstraints.get(0).dim();
        if (initialGuess == null) initialGuess = new ArrayRealVector(n, 0.0);
        double s = Double.NEGATIVE_INFINITY;
        ArrayList<TwiceDifferentiableFunction> iqcs = new ArrayList<TwiceDifferentiableFunction>();
        for (TwiceDifferentiableFunction f: ineqConstraints.toArray(fType)) {
            double y = f.value(initialGuess);
            if (s < y) s = y;
            iqcs.add(new FeasiblePointConstraintFunction(f));
        }
        // If all inequality constraints are < 0, the initial guess is already feasible
        if (s < 0.0) return new PointValuePair(initialGuess.toArray(), s);
        RealVector xStart = initialGuess.append(1.0 + s);
        BarrierOptimizer barrier = new BarrierOptimizer();
        PointValuePair spvp = barrier.optimize(
            new ObjectiveFunction(new FeasiblePointObjectiveFunction(n)),
            new InequalityConstraintSet(iqcs),
            new InitialGuess(xStart.toArray()),
            new HaltingCondition(new NegChecker(n)),
            new InnerOptimizationData(
                new HaltingCondition(new NegChecker(n)))
        );
        double[] xfeas = java.util.Arrays.copyOfRange(spvp.getFirst(), 0, n);
        // return the feasible point (minus augmented dimension s) and final value of s
        return new PointValuePair(xfeas, spvp.getFirst()[n]);
    }

    // known dense Matrix classes
    public static boolean isDense(RealMatrix M) {
        if (M instanceof Array2DRowRealMatrix) return true;
        if (M instanceof BlockRealMatrix) return true;
        return false;
    }
}
