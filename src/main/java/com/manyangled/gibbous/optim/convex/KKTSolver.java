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
import org.apache.commons.math3.optim.OptimizationData;

/**
 * Solves the KKT conditions for {@link NewtonOptimizer}.
 * <p>
 * Default solvers make no assumptions about matrix structure.
 * Custom subclasses can be implemented to take advantage of structural
 * knowledge about matrix sparsity.
 */
public abstract class KKTSolver implements OptimizationData {
    /**
     * solve block factored matrix equation:
     * <pre>
     * | H AT | | v | = -| g |
     * | A  0 | | w |    | h |
     * </pre>
     * <p>
     * where (v, w) are primal/dual delta-x and "nu+" from algorithm 10.2 of <p>
     * Convex Optimization, Boyd and Vandenberghe, Cambridge University Press, 2008.
     * @param H Hessian matrix
     * @param A coefficient matrix of equality constraints
     * @param AT transpose of A
     * @param g gradient, corresponding to H
     * @param h constant vector block corresponding to A
     * @return solution (delta-x, nu+)
     */
    public abstract KKTSolution solve(
        final RealMatrix H,
        final RealMatrix A, final RealMatrix AT,
        final RealVector g, final RealVector h);

    /**
     * Solve constraint-free system Hv = -g <p>
     * returns delta-x (aka v) and lambda-squared from Algorithm 9.5 of <p>
     * Convex Optimization, Boyd and Vandenberghe, Cambridge University Press, 2008. <p>
     * delta-nu (aka w, aka the dual) is returned as null
     * @param H Hessian matrix
     * @param g gradient, corresponding to H
     * @return solution delta-x with lambda-squared
     */
    public abstract KKTSolution solve(final RealMatrix H, final RealVector g);
}
