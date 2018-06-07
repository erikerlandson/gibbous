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

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;

/**
 * A MultivariateFunction that also has a defined gradient and Hessian
 */
public abstract class TwiceDifferentiableFunction implements MultivariateFunction {
    /**
     * Returns the dimensionality of the function domain.
     * If dim() returns (n) then this function expects an n-vector as its input.
     */
    public abstract int dim();
    
    /**
     * Returns the value of this function at (x)
     *
     * @param x a point to evaluate this function at.
     * @return the value of this function at (x)
     */
    public abstract double value(final RealVector x);

    /**
     * Returns the gradient of this function at (x)
     *
     * @param x a point to evaluate this gradient at
     * @return the gradient of this function at (x)
     */
    public abstract RealVector gradient(final RealVector x);

    /**
     * The Hessian of this function at (x)
     *
     * @param x a point to evaluate this Hessian at
     * @return the Hessian of this function at (x)
     */
    public abstract RealMatrix hessian(final RealVector x);
    
    /**
     * Returns the value of this function at (x)
     *
     * @param x a point to evaluate this function at.
     * @return the value of this function at (x)
     */
    @Override
    public double value(final double[] x) {
        return value(new ArrayRealVector(x, false));
    }

    /**
     * Returns the gradient of this function at (x)
     *
     * @param x a point to evaluate this gradient at
     * @return the gradient of this function at (x)
     */
    public RealVector gradient(final double[] x) {
        return gradient(new ArrayRealVector(x, false));
    }

    /**
     * The Hessian of this function at (x)
     *
     * @param x a point to evaluate this Hessian at
     * @return the Hessian of this function at (x)
     */
    public RealMatrix hessian(final double[] x) {
        return hessian(new ArrayRealVector(x, false));
    }
}
