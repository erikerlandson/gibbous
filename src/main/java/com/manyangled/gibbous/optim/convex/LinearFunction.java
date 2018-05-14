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
import org.apache.commons.math3.linear.OpenMapRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;

public class LinearFunction extends TwiceDifferentiableFunction {
    private final RealVector b;
    private final double c;
    private final int n;

    public LinearFunction(RealVector b, double c) {
        int d = b.getDimension();
        if (d < 1) throw new IllegalArgumentException("Dimension must be nonzero");
        this.b = b;
        this.c = c;
        this.n = d;
    }

    public LinearFunction(double[] b, double c) {
        this(new ArrayRealVector(b), c);
    }

    @Override
    public int dim() { return n; }

    @Override
    public double value(final RealVector x) {
        return c + b.dotProduct(x);
    }

    @Override
    public RealVector gradient(final RealVector x) {
        return b.copy();
    }

    @Override
    public RealMatrix hessian(final RealVector x) {
        // the Hessian is just zero for a linear function
        return new OpenMapRealMatrix(n, n);
    }
}
