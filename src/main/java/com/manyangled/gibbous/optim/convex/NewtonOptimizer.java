package com.manyangled.gibbous.optim.convex;

import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.optim.PointValuePair;

// Algorithm 10.2
public class NewtonOptimizer extends ConvexOptimizer {
    private EqualityConstraint eqConstraint;
    private KKTSolver kktSolver = new SchurKKTSolver();

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
        }
    }

    @Override
    public PointValuePair doOptimize() {
        return null;
    }
}
