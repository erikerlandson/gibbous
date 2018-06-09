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

import org.apache.commons.math3.linear.RealVector;

/**
 * This class holds a solution returned from a {@link KKTSolver}.
 * <p>
 * This class is not exposed to the user, and is important only if you are
 * implementing a custom {@link KKTSolver}.
 */
public class KKTSolution {
    /** The delta-x vector solved from the KKT equations */
    public final RealVector xDelta;
    /** The nu+ vector solved from the KKT equations, or null if in (alg 9.5) mode */
    public final RealVector nuPlus;
    /** The lambda-squared value from the KKT equations, or null if in (alg 10.2) mode */
    public final double lambdaSquared;

    /**
     * Construct a KKTSolution for Newton's method with equality constraints.
     * <p>
     * (alg 10.2) of <p>
     * Convex Optimization, Boyd and Vandenberghe, Cambridge University Press, 2008.
     * @param xd the delta-x vector solved from the KKT equation
     * @param nup the nu+ vector solved from the KKT equation
     */
    public KKTSolution(final RealVector xd, final RealVector nup) {
        this.xDelta = xd;
        this.nuPlus = nup;
        this.lambdaSquared = 0.0;
    }

    /**
     * Construct a KKTSolution for Newton's method without equality constraints.
     * <p>
     * (alg 9.5) of <p>
     * Convex Optimization, Boyd and Vandenberghe, Cambridge University Press, 2008.
     * @param xd the delta-x vector solved from the KKT equation
     * @param lsq the lambda-squared value associated with this solution
     */
    public KKTSolution(final RealVector xd, final double lsq) {
        this.xDelta = xd;
        this.lambdaSquared = lsq;
        this.nuPlus = null;
    }
}
