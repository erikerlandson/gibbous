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

import org.apache.commons.math3.util.Pair;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;

/**
 * Given a set of convex functions f[k] for k = 1,2,..., implements the smooth-max
 * over f[k]: (1/a)log(sum(e^(a f[k]))).
 * See also: http://erikerlandson.github.io/blog/2018/05/27/the-gradient-and-hessian-of-the-smooth-max-over-functions/
 */
public class SmoothMaxFunction extends TwiceDifferentiableFunction {
    private final double alpha;
    private final TwiceDifferentiableFunction[] f;

    /**
     * Construct a smooth-max function over a collection of convex functions
     *
     * @param alpha A weighting parameter
     * @param f A collection of convex functions
     */
    public SmoothMaxFunction(double alpha, TwiceDifferentiableFunction... f) {
        if (f.length < 1) throw new IllegalArgumentException("list of functions must be nonempty");
        this.alpha = alpha;
        this.f = f;
    }

    @Override
    public int dim() { return f[0].dim(); }

    @Override
    public double value(final RealVector x) {
        Pair<Double, double[]> pre = precompute(x);
        double z = pre.getFirst();
        double[] exp = pre.getSecond();
        double s = 0.0;
        for (double e: exp) s += e;
        return z + (Math.log(s) / alpha);
    }

    @Override
    public RealVector gradient(final RealVector x) {
        Pair<Double, double[]> pre = precompute(x);
        double[] exp = pre.getSecond();
        RealVector g = new ArrayRealVector(dim(), 0.0);
        double d = 0.0;
        for (int k = 0; k < f.length; ++k) {
            RealVector gk = f[k].gradient(x);
            d += exp[k];
            g.combineToSelf(1.0, exp[k], gk);
        }
        g.mapDivideToSelf(d);
        return g;
    }

    @Override
    public RealMatrix hessian(final RealVector x) {
        Pair<Double, double[]> pre = precompute(x);
        double[] exp = pre.getSecond();
        int n = dim();
        RealMatrix h = new Array2DRowRealMatrix(n, n);
        double d = 0.0;
        for (int k = 0; k < f.length; ++k) {
            d += exp[k];
            RealVector gfk = f[k].gradient(x);
            RealMatrix hfk = f[k].hessian(x);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) {
                    h.addToEntry(i, j, exp[k] * hfk.getEntry(i, j));
                    h.addToEntry(i, j, alpha * exp[k] * gfk.getEntry(i) * gfk.getEntry(j));
                }
        }
        d = 1.0 / d;
        RealVector g = gradient(x);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                h.multiplyEntry(i, j, d);
                h.addToEntry(i, j, (-alpha) * g.getEntry(i) * g.getEntry(j));
            }
        return h;
    }

    // Pre-computes z, the maximum of f[k](x), and also e^(α(f[k]-z)) for each f[k].
    // See: http://erikerlandson.github.io/blog/2018/05/28/computing-smooth-max-and-its-gradients-without-over-and-underflow/
    private Pair<Double, double[]> precompute(final RealVector x) {
        double[] exp = new double[f.length];
        double z = Double.NEGATIVE_INFINITY;
        for (int k = 0; k < f.length; ++k) {
            double fk = f[k].value(x);
            exp[k] = fk;
            z = Math.max(z, fk);
        }
        for (int k = 0; k < f.length; ++k) {
            double fk = exp[k];
            exp[k] = Math.exp(alpha * (fk - z));
        }
        return new Pair<Double, double[]>(z, exp);
    }
}
