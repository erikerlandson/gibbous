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

import org.apache.commons.math3.util.Pair;
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;

/**
 * An optimizer that can minimize a convex function in the presence of a set of
 * convex inequality constraints and linear equality constraints.
 * <p>
 * An implementation of the Barrier Method (Algorithm 11.1) from
 * Convex Optimization, Boyd and Vandenberghe, Cambridge University Press, 2008.
 * <p>
 * {@link BarrierOptimizer} supports the following {@link OptimizationData} parameters as arguments
 * to {@link #optimize(OptimizationData...)}:
 * <ul>
 *   <li>convex objective function: {@link ObjectiveFunction} - mandatory: must contain a {@link TwiceDifferentiableFunction} </li>
 *   <li>initial guess: {@link InitialGuess} - mandatory: must be strictly feasible w.r.t. all inequality constraints. </li>
 *   <li>convex inequality constraints: {@link InequalityConstraintSet} - optional </li>
 *   <li>linear inequality constraints: {@link LinearInequalityConstraint} - optional </li>
 *   <li>linear equality constraints: {@link LinearEqualityConstraint} - optional </li>
 *   <li>convergence epsilon: {@link ConvergenceEpsilon} - optional </li>
 *   <li>inner optimizer parameters: {@link InnerOptimizationData} - optional: passed down to {@link NewtonOptimizer} inner calls. </li>
 * </ul>
 * <p>
 * NOTE: all parameters to {@link #optimize(OptimizationData...)} are also passed to {@link NewtonOptimizer}, and so
 * for example setting {@link ConvergenceEpsilon} here will also set it for inner calls to {@link NewtonOptimizer}. However, any
 * settings passed via {@link InnerOptimizationData} are applied last for {@link NewtonOptimizer}, and so will have precedence.
 */
public class BarrierOptimizer extends ConvexOptimizer {
    private ArrayList<TwiceDifferentiableFunction> constraintFunctions =
        new ArrayList<TwiceDifferentiableFunction>();
    private RealVector xStart;
    private double epsilon = 1e-10;
    private double mu = BarrierMu.BARRIER_MU_DEFAULT;
    private double t0 = BarrierMu.BARRIER_T0_DEFAULT;
    private OptimizationData[] odType = new OptimizationData[0];
    private HaltingCondition halting;
    private ArrayList<OptimizationData> newtonArgs = new ArrayList<OptimizationData>();
    private ArrayList<OptimizationData> innerArgs = new ArrayList<OptimizationData>();

    public BarrierOptimizer() {
        super();
    }

    @Override
    public PointValuePair optimize(OptimizationData... optData) {
        return super.optimize(optData);
    }

    private boolean canPassFromMain(OptimizationData data) {
        if (data instanceof ObjectiveFunction) return false;
        if (data instanceof InitialGuess) return false;
        if (data instanceof HaltingCondition) return false;
        return true;
    }

    private boolean canPassFromInner(OptimizationData data) {
        if (data instanceof HaltingCondition) return true;
        return canPassFromMain(data);
    }

    @Override
    protected void parseOptimizationData(OptimizationData... optData) {
        super.parseOptimizationData(optData);
        // save these for configuring newton optimizers
        for (OptimizationData data: optData) {
            if (canPassFromMain(data)) {
                newtonArgs.add(data);
            }
            if (data instanceof ConvergenceEpsilon) {
                epsilon = ((ConvergenceEpsilon)data).epsilon;
                continue;
            }
            if (data instanceof BarrierMu) {
                mu = ((BarrierMu)data).mu;
                t0 = ((BarrierMu)data).t0;
            }
            if (data instanceof LinearInequalityConstraint) {
                for (TwiceDifferentiableFunction f: ((LinearInequalityConstraint)data).lcf)
                    constraintFunctions.add(f);
                continue;
            }
            if (data instanceof InequalityConstraintSet) {
                constraintFunctions.addAll(((InequalityConstraintSet)data).constraints);
                continue;
            }
            if (data instanceof HaltingCondition) {
                halting = (HaltingCondition)data;
                continue;
            }
            if (data instanceof InnerOptimizationData) {
                for (OptimizationData d: ((InnerOptimizationData)data).optData.toArray(odType))
                    if (canPassFromInner(d)) innerArgs.add(d);
                continue;
            }
        }
        // if we got here, convexObjective exists
        int n = convexObjective.dim();
        if (this.getStartPoint() != null) {
            xStart = new ArrayRealVector(this.getStartPoint());
            if (xStart.getDimension() != n)
                throw new DimensionMismatchException(xStart.getDimension(), n);
        } else {
            xStart = new ArrayRealVector(n, 0.0);
        }
        // append any "inner" args - this overrides anything currently in newtonArgs
        newtonArgs.addAll(innerArgs);
    }

    @Override
    public PointValuePair doOptimize() {
        double m = (double)((constraintFunctions != null) ? constraintFunctions.size() : 0);
        if (m == 0.0) {
            // if there are no inequality constraints, invoke newton's method directly
            ArrayList<OptimizationData> args = (ArrayList<OptimizationData>)newtonArgs.clone();
            args.add(new ObjectiveFunction(convexObjective));
            args.add(new InitialGuess(xStart.toArray()));
            NewtonOptimizer newton = new NewtonOptimizer();
            return newton.optimize(args.toArray(odType));
        }
        RealVector x = xStart;
        for (double t = t0; (t * epsilon) <= m ; t *= mu) {
            TwiceDifferentiableFunction bf = new LogBarrierFunction(t, convexObjective, constraintFunctions);
            NewtonOptimizer newton = new NewtonOptimizer();
            ArrayList<OptimizationData> args = (ArrayList<OptimizationData>)newtonArgs.clone();
            args.add(new ObjectiveFunction(bf));
            args.add(new InitialGuess(x.toArray()));
            PointValuePair pvp = newton.optimize(args.toArray(odType));
            // update for next iteration
            RealVector xprv = x;
            x = new ArrayRealVector(pvp.getFirst());
            if ((halting != null) && halting.checker.converged(
                    getIterations(),
                    new Pair<RealVector, Double>(xprv, convexObjective.value(xprv)),
                    new Pair<RealVector, Double>(x, convexObjective.value(x)))) {
                break;
            }
        }
        return new PointValuePair(x.toArray(), convexObjective.value(x));
    }
}
