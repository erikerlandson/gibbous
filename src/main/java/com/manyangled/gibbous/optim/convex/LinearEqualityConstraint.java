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
 * Represents a set of linear equality constraints given as Ax = b.
 */
public class LinearEqualityConstraint implements OptimizationData {
    public final RealMatrix A;
    public final RealVector b;

    /**
     * Construct a set of linear equality constraints Ax = b.
     * Represents equations A[i].x = b[i], for each row of A.
     * @param A the matrix of linear weights
     * @param b the vector of constants
     */
    public LinearEqualityConstraint(final RealMatrix A, final RealVector b) {
        int k = A.getRowDimension();
        if (b.getDimension() != k)
            throw new DimensionMismatchException(b.getDimension(), k);
        this.A = A;
        this.b = b;
    }

    /**
     * Construct a set of linear equality constraints Ax = b.
     * Represents equations A[i].x = b[i], for each row of A.
     * @param A the matrix of linear weights
     * @param b the vector of constants
     */    
    public LinearEqualityConstraint(final double[][] A, final double[] b) {
        this(new Array2DRowRealMatrix(A), new ArrayRealVector(b));
    }
}
