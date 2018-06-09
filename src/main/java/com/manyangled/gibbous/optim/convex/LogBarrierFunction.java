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

import java.lang.Math;
import java.util.Collection;
import java.util.Iterator;
import java.util.Arrays;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.exception.DimensionMismatchException;

/**
 * Given a convex objective function f0 and convex constraints f[k],
 * computes the log barrier function: <p>
 * b(x) = (t)f0(x) - sum(log(f[k](x))) <p>
 * returns +inf if any f[k](x) &gt;= 0
 */
public class LogBarrierFunction extends TwiceDifferentiableFunction {
    private final double t;
    private final TwiceDifferentiableFunction f0;
    private final TwiceDifferentiableFunction[] f;
    private final int n;

    /**
     * construct a log-barrier function b(x) = (t)f0(x) - sum(log(f[k](x)))
     * @param t multiplier constant for f0, must be &gt; 0
     * @param f0 a convex objective function
     * @param f a list of convex constraint functions
     */
    public LogBarrierFunction(double t, TwiceDifferentiableFunction f0, TwiceDifferentiableFunction[] f) {
        if (t <= 0.0) throw new IllegalArgumentException("t must be > 0");
        this.t = t;
        this.f0 = f0;
        this.n = f0.dim();
        this.f = f;
        for (TwiceDifferentiableFunction fi: f) {
            if (fi.dim() != n) throw new DimensionMismatchException(fi.dim(), n);
        }
    }

    /**
     * construct a log-barrier function b(x) = (t)f0(x) - sum(log(f[k](x)))
     * @param t multiplier constant for f0, must be &gt; 0
     * @param f0 a convex objective function
     * @param f a list of convex constraint functions
     */
    public LogBarrierFunction(double t, TwiceDifferentiableFunction f0, Collection<TwiceDifferentiableFunction> f) {
        this(t, f0, f.toArray(new TwiceDifferentiableFunction[0]));
    }

    @Override
    public int dim() {
        return n;
    }

    @Override
    public double value(final RealVector x) {
        double v = t * f0.value(x);
        for (TwiceDifferentiableFunction fi: f) {
            double ti = fi.value(x);
            if (ti >= 0.0) {
                return Double.POSITIVE_INFINITY;
            }
            v -= Math.log(-ti);
        }
        return v;
    }

    @Override
    public RealVector gradient(final RealVector x) {
        // g should be dense, due to contributions of barrier functions
        RealVector g = new ArrayRealVector(f0.gradient(x).toArray(), false);
        g.mapMultiplyToSelf(t);
        for (TwiceDifferentiableFunction fi: f) {
            double zi = -1.0 / fi.value(x);
            g.combineToSelf(1.0, zi, fi.gradient(x));
        }
        return g;
    }

    @Override
    public RealMatrix hessian(final RealVector x) {
        // h should be dense, due to contributions of barrier functions
        RealMatrix h = new Array2DRowRealMatrix(f0.hessian(x).getData(), false);
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                h.multiplyEntry(j, k, t);
        for (TwiceDifferentiableFunction fi: f) {
            double vi = fi.value(x);
            RealVector gi = fi.gradient(x);
            RealMatrix hi = fi.hessian(x);
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                    h.addToEntry(j, k, (gi.getEntry(j)*gi.getEntry(k)/(vi*vi)) - hi.getEntry(j, k)/vi);
                }
            }
        }
        return h;
    }
}
