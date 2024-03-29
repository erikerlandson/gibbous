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

import org.apache.commons.math3.optim.OptimizationData;

/**
 * Defines an epsilon as a threshold for measuring a small change that approximates convergences of
 * some iterative optimization.
 */
public class ConvergenceEpsilon implements OptimizationData {
    /**
     * The actual epsilon value. Strictly &gt; 0.
     */
    public final double epsilon;

    /**
     * Construct a new ConvergenceEpsilon parameter with the given epsilon value
     *
     * @param eps The epsilon value to use. Must be strictly &gt; 0.
     */
    public ConvergenceEpsilon(double eps) {
        if (eps <= 0.0)
            throw new IllegalArgumentException("epsilon must be > 0");
        this.epsilon = eps;
    }

    /** Default convergence epsilon value */
    public static final double CONVERGENCE_EPSILON_DEFAULT = 1e-9;
}
