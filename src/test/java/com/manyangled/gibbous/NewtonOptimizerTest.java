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
import com.manyangled.gibbous.optim.convex.EqualityConstraint;

import static com.manyangled.gibbous.COTestingUtils.translatedQF;
import static com.manyangled.gibbous.COTestingUtils.eps;

public class NewtonOptimizerTest {
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

    @Test
    public void testSimpleConstrained2D() {
        double[] center = { 0.0, 0.0 };
        double h = 0.0;
        QuadraticFunction q = translatedQF(h, center);
        double[] ig = { 10.0, 10.0 };
        double[][] A = { { 1.0, 1.0 } }; // equality constraint x + y = 1
        double[] b = { 1.0 };
        double[] xminTarget = { 0.5, 0.5 };
        double vminTarget = 0.25;
        NewtonOptimizer nopt = new NewtonOptimizer();
        PointValuePair pvp = nopt.optimize(
            new ObjectiveFunction(q),
            new EqualityConstraint(A, b),
            new InitialGuess(ig));
        double[] xmin = pvp.getFirst();
        double vmin = pvp.getSecond();
        assertArrayEquals(xminTarget, xmin, eps);
        assertEquals(vminTarget, vmin, eps);
    }

    @Test
    public void testTranslatedConstrained2D() {
        double[] center = { 3.0, 3.0 };
        double h = 7.0;
        QuadraticFunction q = translatedQF(h, center);
        double[] ig = { 10.0, 10.0 };
        double[][] A = { { 1.0, 1.0 } }; // equality constraint x + y = 7
        double[] b = { 7.0 };
        double[] xminTarget = { 3.5, 3.5 };
        double vminTarget = 7.25;
        NewtonOptimizer nopt = new NewtonOptimizer();
        PointValuePair pvp = nopt.optimize(
            new ObjectiveFunction(q),
            new EqualityConstraint(A, b),
            new InitialGuess(ig));
        double[] xmin = pvp.getFirst();
        double vmin = pvp.getSecond();
        assertArrayEquals(xminTarget, xmin, eps);
        assertEquals(vminTarget, vmin, eps);
    }

    @Test
    public void testSimpleConstrained3D() {
        double[] center = { 0.0, 0.0, 0.0 };
        double h = 0.0;
        QuadraticFunction q = translatedQF(h, center);
        double[][] A = { { 1.0, 1.0, 1.0 } }; // equality constraint x + y + z = 1
        double[] b = { 1.0 };
        double[] xminTarget = { 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 };
        double vminTarget = 1.0 / 6.0;
        NewtonOptimizer nopt = new NewtonOptimizer();
        PointValuePair pvp = nopt.optimize(
            new ObjectiveFunction(q),
            new EqualityConstraint(A, b));
        double[] xmin = pvp.getFirst();
        double vmin = pvp.getSecond();
        assertArrayEquals(xminTarget, xmin, eps);
        assertEquals(vminTarget, vmin, eps);
    }

    @Test
    public void testTranslatedConstrained3D() {
        double[] center = { 10.0, 10.0, 10.0 };
        double h = -100.0;
        QuadraticFunction q = translatedQF(h, center);
        double[][] A = { { 1.0, 1.0, 1.0 } }; // equality constraint x + y + z = 31
        double[] b = { 31.0 };
        double[] xminTarget = { 10.0 + (1.0 / 3.0), 10.0 + (1.0 / 3.0), 10.0 + (1.0 / 3.0) };
        double vminTarget = -100.0 + (1.0 / 6.0);
        NewtonOptimizer nopt = new NewtonOptimizer();
        PointValuePair pvp = nopt.optimize(
            new ObjectiveFunction(q),
            new EqualityConstraint(A, b));
        double[] xmin = pvp.getFirst();
        double vmin = pvp.getSecond();
        assertArrayEquals(xminTarget, xmin, eps);
        assertEquals(vminTarget, vmin, eps);
    }
}
