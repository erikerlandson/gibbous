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
import org.apache.commons.math3.exception.DimensionMismatchException;

public class LogBarrierFunction extends ConvexFunction {

    private final double t;
    private final ConvexFunction f0;
    private final ConvexFunction[] f;
    private final int n;

    public LogBarrierFunction(double t, ConvexFunction f0, ConvexFunction[] f) {
        this.t = t;
        this.f0 = f0;
        this.n = f0.dim();
        this.f = f;
        for (ConvexFunction fi: f) {
            if (fi.dim() != n) throw new DimensionMismatchException(fi.dim(), n);
        }
    }

    public LogBarrierFunction(double t, ConvexFunction f0, Collection<ConvexFunction> cf) {
        this.t = t;
        this.f0 = f0;
        this.n = f0.dim();
        int m = cf.size();
        f = new ConvexFunction[m];
        Iterator e = cf.iterator();
        int j = 0;
        while (e.hasNext()) {
            f[j] = (ConvexFunction)(e.next());
            if (f[j].dim() != n) throw new DimensionMismatchException(f[j].dim(), n);
            j += 1;
        }
    }

    @Override
    public int dim() {
        return n;
    }

    @Override
    public double value(final RealVector x) {
        double v = t * f0.value(x);
        for (ConvexFunction fi: f) {
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
        RealVector g = f0.gradient(x);
        g.mapMultiplyToSelf(t);
        for (ConvexFunction fi: f) {
            double zi = -1.0 / fi.value(x);
            g.combineToSelf(1.0, zi, fi.gradient(x));
        }
        return g;
    }

    @Override
    public RealMatrix hessian(final RealVector x) {
        RealMatrix h = f0.hessian(x).scalarMultiply(t);
        for (ConvexFunction fi: f) {
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
