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

package com.manyangled.gibbous;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertArrayEquals;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.DiagonalMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;

import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.InitialGuess;

import com.manyangled.gibbous.optim.convex.NewtonOptimizer;
import com.manyangled.gibbous.optim.convex.QuadraticFunction;

public class NewtonOptimizerTest {
    private double eps = 1e-9;

    private QuadraticFunction translatedQF(double h, double[] center) {
        double[] all1 = new double[center.length];
        java.util.Arrays.fill(all1, 1.0);
        RealVector cv = new ArrayRealVector(center);
        RealMatrix A = new DiagonalMatrix(all1);
        RealVector b = cv.mapMultiply(-1.0);
        double c = h + 0.5*cv.dotProduct(cv);
        return new QuadraticFunction(A, b, c);
    }

    @Test
    public void testSimpleQuadratic2D() {
        double[] center = { 0.0, 0.0 };
        double h = 0.0;
        QuadraticFunction q = translatedQF(h, center);
        double[] ig = { 10.0, 10.0 };
        NewtonOptimizer nopt = new NewtonOptimizer();
        PointValuePair pvp = nopt.optimize(
            new ObjectiveFunction(q),
            new InitialGuess(ig));
        double[] xmin = pvp.getFirst();
        double vmin = pvp.getSecond();
        assertArrayEquals(center, xmin, eps);
        assertEquals(h, vmin, eps);
    }

    @Test
    public void testTranslatedQuadratic2D() {
        double[] center = { 3.0, 7.0 };
        double h = 11.0;
        QuadraticFunction q = translatedQF(h, center);
        NewtonOptimizer nopt = new NewtonOptimizer();
        PointValuePair pvp = nopt.optimize(
            new ObjectiveFunction(q));
        double[] xmin = pvp.getFirst();
        double vmin = pvp.getSecond();
        assertArrayEquals(center, xmin, eps);
        assertEquals(h, vmin, eps);        
    }

    @Test
    public void testTranslatedQuadratic3D() {
        double[] center = { 3.0, 7.0, 11.0 };
        double h = -13.0;
        QuadraticFunction q = translatedQF(h, center);
        NewtonOptimizer nopt = new NewtonOptimizer();
        PointValuePair pvp = nopt.optimize(
            new ObjectiveFunction(q));
        double[] xmin = pvp.getFirst();
        double vmin = pvp.getSecond();
        assertArrayEquals(center, xmin, eps);
        assertEquals(h, vmin, eps);        
    }
}
