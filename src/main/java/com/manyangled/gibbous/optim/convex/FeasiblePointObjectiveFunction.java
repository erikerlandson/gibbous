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
import org.apache.commons.math3.linear.OpenMapRealMatrix;

public class FeasiblePointObjectiveFunction extends TwiceDifferentiableFunction {
    private final int n;

    public FeasiblePointObjectiveFunction(int n) {
        this.n = n;
    }

    @Override
    public int dim() { return 1 + n; }

    @Override
    public double value(final RealVector x) {
        return x.getEntry(n);
    }

    @Override
    public RealVector gradient(final RealVector x) {
        RealVector g = new ArrayRealVector(1 + n);
        g.set(0.0);
        g.setEntry(n, 1.0);
        return g;
    }

    @Override
    public RealMatrix hessian(final RealVector x) {
        return new OpenMapRealMatrix(1 + n, 1 + n);
    }
}
