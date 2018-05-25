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

public class BarrierOptimizer extends ConvexOptimizer {
    private ArrayList<TwiceDifferentiableFunction> constraintFunctions =
        new ArrayList<TwiceDifferentiableFunction>();
    private RealVector xStart;
    private double epsilon = 1e-10;
    private double mu = 10.0;
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
            if (data instanceof LinearInequalityConstraint) {
                RealMatrix A = ((LinearInequalityConstraint)data).A;
                RealVector b = ((LinearInequalityConstraint)data).b;
                for (int k = 0; k < b.getDimension(); ++k) {
                    constraintFunctions.add(new LinearFunction(A.getRowVector(k), b.getEntry(k)));
                }
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
        for (double t = 1.0 ; (t * epsilon) <= m ; t *= mu) {
            TwiceDifferentiableFunction bf = new LogBarrierFunction(t, convexObjective, constraintFunctions);
                //RealMatrix th = bf.hessian(x);
                //System.out.format("*** t= %f  x= %s  h= %s\n", t, x, th);
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
