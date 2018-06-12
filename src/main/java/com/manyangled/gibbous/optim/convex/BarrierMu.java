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
 * The mu parameter for the Barrier Method used by {@link BarrierOptimizer}.
 * This is the scaling factor for the objective function multiplier (t),
 * as described in (Algorithm 11.1) from <p>
 * Convex Optimization, Boyd and Vandenberghe, Cambridge University Press, 2008.
 */
public class BarrierMu implements OptimizationData {
    /** The value of mu, the scaling factor for the objective function multiplier */
    public final double mu;
    /** The initial value for t, the objective function multiplier */
    public final double t0;

    /**
     * Construct a Barrier Method mu parameter.
     * @param mu the value of the parameter mu, the scaling factor for the objective function multiplier t. Must be &gt; 1.
     * @param t0 the initial value for t, the objective function multiplier. Must be &gt; 0.
     */
    public BarrierMu(double mu, double t0) {
        if (mu <= 1.0) throw new IllegalArgumentException("mu must be > 1");
        if (t0 <= 0.0) throw new IllegalArgumentException("t0 must be > 0");
        this.mu = mu;
        this.t0 = t0;
    }

    /**
     * Construct a Barrier Method mu parameter, the scaling factor for the objective function multiplier t.
     * @param mu the value of the parameter. Must be &gt; 1.
     */
    public BarrierMu(double mu) {
        this(mu, BARRIER_T0_DEFAULT);
    }

    /** Default value for mu */
    public static final double BARRIER_MU_DEFAULT = 15.0;
    /** Default initial value for (t) */
    public static final double BARRIER_T0_DEFAULT = 1.0;
}
