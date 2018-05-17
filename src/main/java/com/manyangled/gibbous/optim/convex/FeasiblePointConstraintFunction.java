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
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;

public class FeasiblePointConstraintFunction extends TwiceDifferentiableFunction {
    private final TwiceDifferentiableFunction f;
    private final int n;

    public FeasiblePointConstraintFunction(TwiceDifferentiableFunction f) {
        this.f = f;
        this.n = f.dim();
    }

    @Override
    public int dim() { return 1 + n; }

    @Override
    public double value(final RealVector x) {
        return f.value(x.getSubVector(0, n)) - x.getEntry(n);
    }

    @Override
    public RealVector gradient(final RealVector x) {
        RealVector g = new ArrayRealVector(1 + n);
        g.setSubVector(0, f.gradient(x.getSubVector(0, n)));
        g.setEntry(n, -1.0);
        return g;
    }

    @Override
    public RealMatrix hessian(final RealVector x) {
        RealMatrix fh = f.hessian(x.getSubVector(0, n));
        double[][] h = new double[1 + n][1 + n];
        fh.copySubMatrix(0, n - 1, 0, n - 1, h);
        for (int j = 0; j <= n; ++j) {
            h[j][n] = 0.0;
            h[n][j] = 0.0;
        }
        return new Array2DRowRealMatrix(h, false);
    }
}