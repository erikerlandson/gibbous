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
 * The backtracking beta parameter, used in {@link NewtonOptimizer}.
 * As described in (Algorithm 9.2) of
 * Convex Optimization, Boyd and Vandenberghe, Cambridge University Press, 2008.
 */
public class BacktrackBeta implements OptimizationData {
    public final double beta;

    /**
     * Construct a backtracking beta parameter
     *
     * @param b the value for beta. Must be on interval (0,1)
     */
    public BacktrackBeta(double b) {
        if ((b <= 0.0) || (b >= 1.0))
            throw new IllegalArgumentException("beta must be on (0,1)");
        this.beta = b;
    }

    public static final double BACKTRACK_BETA_DEFAULT = 0.8;
}
