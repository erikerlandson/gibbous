package com.manyangled.gibbous.optim.convex;

import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.nonlinear.scalar.MultivariateOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.analysis.MultivariateFunction;

public abstract class ConvexOptimizer extends MultivariateOptimizer {
    protected ConvexFunction convexObjective;

    protected ConvexOptimizer() {
        super(null);
    }

    @Override
    protected void parseOptimizationData(OptimizationData... optData) {
        super.parseOptimizationData(optData);
        for (OptimizationData data: optData) {
            if (data instanceof ObjectiveFunction) {
                MultivariateFunction f = ((ObjectiveFunction)data).getObjectiveFunction();
                if (f instanceof ConvexFunction) {
                  convexObjective = (ConvexFunction)f;
                } else {
                    throw new IllegalArgumentException("ConvexFunction objective required");
                }
                continue;
            }
        }
        if (convexObjective == null)
            throw new IllegalStateException("Expected a ConvexFunction argument");
    }

    @Override
    public PointValuePair optimize(OptimizationData... optData) {
        return super.optimize(optData);
    }
}
