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
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.DiagonalMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.exception.DimensionMismatchException;

/**
 * Given A, b, c, implements 0.5*(x^T)A(x) + b.x + c;
 * The gradient is A(x) + b, and the Hessian is A
 */
public class QuadraticFunction extends TwiceDifferentiableFunction {
    private final RealMatrix A;
    private final RealVector b;
    private final double c;
    private final int n;

    /**
     * Construct quadratic function 0.5*(x^T)A(x) + b.x + c
     *
     * @param A square matrix of weights for quadratic terms.
     * Typically expected to be positive definite or positive semi-definite.
     * @param b vector of weights for linear terms.
     * @param c a constant
     */
    public QuadraticFunction(RealMatrix A, RealVector b, double c) {
        int d = b.getDimension();
        if (d < 1) throw new IllegalArgumentException("Dimension must be nonzero");
        if (A.getRowDimension() != d)
            throw new DimensionMismatchException(A.getRowDimension(), d);
        MatrixUtils.checkSymmetric(A, 1e-6);
        this.A = A.copy();
        this.b = b.copy();
        this.c = c;
        this.n = d;
    }

    /**
     * Construct quadratic function 0.5*(x^T)A(x) + b.x + c
     *
     * @param A square matrix of weights for quadratic terms.
     * Typically expected to be positive definite or positive semi-definite.
     * @param b vector of weights for linear terms.
     * @param c a constant
     */
    public QuadraticFunction(double[][] A, double[] b, double c) {
        this(new Array2DRowRealMatrix(A), new ArrayRealVector(b), c);
    }

    @Override
    public int dim() { return n; }

    @Override
    public double value(final RealVector x) {
        double v = 0.5 * (A.operate(x)).dotProduct(x);
        v += b.dotProduct(x);
        v += c;
        return v;
    }

    @Override
    public RealVector gradient(final RealVector x) {
        return (A.operate(x)).add(b);
    }

    @Override
    public RealMatrix hessian(final RealVector x) {
        return A.copy();
    }

    /**
     * Create a quadratic function that corresponds to s((x-c).(x-c) &lt; r^2).
     * That is, constrained to an n-dimensional ball of radius r, with scaling factor s.
     *
     * @param center the center of the n-ball
     * @param r a radius, &gt; 0
     * @param s a scaling constant, &gt; 0
     * @return quadratic constraint function for the n-ball constraint
     */
    public static QuadraticFunction nBallConstraintFunction(RealVector center, double r, double s) {
        int n = center.getDimension();
        if (n < 1) throw new IllegalArgumentException("center vector must have dimension > 0");
        if (s <= 0.0) throw new IllegalArgumentException("scale s must be > 0");
        if (r <= 0.0) throw new IllegalArgumentException("radius r must be > 0");
        double[] alls = new double[n];
        java.util.Arrays.fill(alls, s);
        RealMatrix A = new DiagonalMatrix(alls);
        RealVector b = center.mapMultiply(-s);
        double c = 0.5 * ((s * center.dotProduct(center)) - (r * r));
        return new QuadraticFunction(A, b, c);
    }

    /**
     * Create a quadratic function that corresponds to s((x-c).(x-c) &lt; r^2).
     * That is, constrained to an n-dimensional ball of radius r, with scaling factor s.
     *
     * @param center the center of the n-ball
     * @param r a radius, &gt; 0
     * @param s a scaling constant, &gt; 0
     * @return quadratic constraint function for the n-ball constraint
     */
    public static QuadraticFunction nBallConstraintFunction(double[] center, double r, double s) {
        return nBallConstraintFunction(new ArrayRealVector(center, false), r, s);
    }

    /**
     * Create a quadratic function that corresponds to (x-c).(x-c) &lt; r^2.
     * That is, constrained to an n-dimensional ball of radius r
     *
     * @param center the center of the n-ball
     * @param r a radius, &gt; 0
     * @return quadratic constraint function for the n-ball constraint
     */
    public static QuadraticFunction nBallConstraintFunction(RealVector center, double r) {
        return nBallConstraintFunction(center, r, 1.0);
    }

    /**
     * Create a quadratic function that corresponds to (x-c).(x-c) &lt; r^2.
     * That is, constrained to an n-dimensional ball of radius r
     *
     * @param center the center of the n-ball
     * @param r a radius, &gt; 0
     * @return quadratic constraint function for the n-ball constraint
     */
    public static QuadraticFunction nBallConstraintFunction(double[] center, double r) {
        return nBallConstraintFunction(new ArrayRealVector(center), r, 1.0);
    }    
}
