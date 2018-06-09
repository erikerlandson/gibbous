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

import java.util.ArrayList;
import java.util.Collection;

import org.apache.commons.math3.optim.OptimizationData;

/**
 * A collection of convex constraint functions. A constraint f is satisfied by x if
 * and only if f(x) &lt; 0. Constraints contained in this object are appended to the
 * list of constraints applied by a convex optimizer. Currently used by
 * {@link BarrierOptimizer} and {@link ConvexOptimizer#feasiblePoint(OptimizationData... optData)}
 */
public class InequalityConstraintSet implements OptimizationData {
    public final ArrayList<TwiceDifferentiableFunction> constraints =
        new ArrayList<TwiceDifferentiableFunction>();

    /**
     * Construct an inequality constraint set from a Collection of convex constraint functions
     * @param constraints the Collection of constraint functions to apply
     */
    public InequalityConstraintSet(Collection<TwiceDifferentiableFunction> constraints) {
        this.constraints.addAll(constraints);
    }

    /**
     * Construct an inequality constraint set from an argument list (or array) of convex
     * constraint functions
     * @param constraints the list of constraint functions to apply
     */
    public InequalityConstraintSet(TwiceDifferentiableFunction... constraints) {
        for (TwiceDifferentiableFunction f: constraints)
            this.constraints.add(f);
    }
}
