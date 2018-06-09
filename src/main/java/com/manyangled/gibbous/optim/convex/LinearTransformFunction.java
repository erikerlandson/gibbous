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

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import static com.manyangled.gibbous.optim.convex.ConvexOptimizer.isDense;

/**
 * Applies a linear transform to a function, from f(x) to t(x) = af(x) + b
 */
public class LinearTransformFunction extends TwiceDifferentiableFunction {
    private final double a;
    private final double b;
    private final TwiceDifferentiableFunction f;

    /**
     * construct a linear transform, af(x) + b, of a function f(x)
     * @param a the linear coefficient
     * @param b a constant
     * @param f the function
     */
    public LinearTransformFunction(
        final double a, final double b,
        final TwiceDifferentiableFunction f) {
        this.a = a;
        this.b = b;
        this.f = f;
    }

    @Override
    public int dim() { return f.dim(); }

    @Override
    public double value(final RealVector x) {
        return b + (a * f.value(x));
    }

    @Override
    public RealVector gradient(final RealVector x) {
        RealVector g = f.gradient(x);
        if (a != 1.0) g.mapMultiplyToSelf(a);
        return g;
    }

    @Override
    public RealMatrix hessian(final RealVector x) {
        RealMatrix h = f.hessian(x);
        if (a == 1.0) return h;
        if (isDense(h)) {
            int n = h.getRowDimension();
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k)
                    h.multiplyEntry(j, k, a);
            return h;
        } else {
            return h.scalarMultiply(a);
        }
    }
}
