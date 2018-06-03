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
import static org.junit.Assert.*;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.DiagonalMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;

import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.InitialGuess;

import com.manyangled.gibbous.optim.convex.TwiceDifferentiableFunction;
import com.manyangled.gibbous.optim.convex.QuadraticFunction;
import com.manyangled.gibbous.optim.convex.LinearFunction;
import com.manyangled.gibbous.optim.convex.LinearEqualityConstraint;
import com.manyangled.gibbous.optim.convex.InequalityConstraintSet;

import com.manyangled.gibbous.optim.convex.ConvexOptimizer;

import static com.manyangled.gibbous.COTestingUtils.translatedQF;
import static com.manyangled.gibbous.COTestingUtils.eps;

public class FeasiblePointTest {
    private void testFeasibleConstraints(InequalityConstraintSet constraints) {
        PointValuePair pvp = ConvexOptimizer.feasiblePoint(constraints);
        //System.out.format("fkmax= %s\n", pvp.getSecond());
        assertTrue(pvp.getSecond() < 0.0);
        for (TwiceDifferentiableFunction f: constraints.constraints) {
            double y = f.value(pvp.getFirst());
            //System.out.format("y= %s\n", y);
            assertTrue(f.value(pvp.getFirst()) < 0.0);
        }
    }

    @Test
    public void test2DHalfPlane() {
        InequalityConstraintSet hp = new InequalityConstraintSet(
            new LinearFunction(new double[] { -1.0, 0.0 }, 100.0)
        );
        testFeasibleConstraints(hp);
    }


    @Test
    public void test2DSquareRegion() {
        InequalityConstraintSet hp = new InequalityConstraintSet(
            new LinearFunction(new double[] { 1.0, 0.0 }, -5.0),
            new LinearFunction(new double[] { -1.0, 0.0 }, 1.0),
            new LinearFunction(new double[] { 0.0, 1.0 }, -5.0),
            new LinearFunction(new double[] { 0.0, -1.0 }, 1.0)
        );
        testFeasibleConstraints(hp);
    }

}
