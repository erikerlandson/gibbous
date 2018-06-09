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
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.exception.DimensionMismatchException;

/**
 * A set of linear inequality constraints expressed as Ax &lt; b
 */
public class LinearInequalityConstraint implements OptimizationData {
    /** The corresponding set of individual linear constraint functions */
    public final LinearFunction[] lcf;

    /**
     * Construct a set of linear inequality constraints from Ax &lt; B
     * @param A A matrix linear coefficient vectors
     * @param b A vector of constants
     */
    public LinearInequalityConstraint(final RealMatrix A, final RealVector b) {
        int k = A.getRowDimension();
        if (b.getDimension() != k)
            throw new DimensionMismatchException(b.getDimension(), k);
        this.lcf = new LinearFunction[k];
        for (int j = 0; j < k; ++j)
            lcf[j] = new LinearFunction(A.getRowVector(j), -b.getEntry(j));
    }

    /**
     * Construct a set of linear inequality constraints from Ax &lt; B
     * @param A A matrix linear coefficient vectors
     * @param b A vector of constants
     */
    public LinearInequalityConstraint(final double[][] A, final double[] b) {
        this(new Array2DRowRealMatrix(A), new ArrayRealVector(b));
    }
}
