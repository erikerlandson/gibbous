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

public class InequalityConstraintSet implements OptimizationData {
    public final ArrayList<TwiceDifferentiableFunction> constraints =
        new ArrayList<TwiceDifferentiableFunction>();

    public InequalityConstraintSet(Collection<TwiceDifferentiableFunction> constraints) {
        this.constraints.addAll(constraints);
    }

    public InequalityConstraintSet(TwiceDifferentiableFunction[] constraints) {
        for (TwiceDifferentiableFunction f: constraints)
            this.constraints.add(f);
    }

    public void add(TwiceDifferentiableFunction f) {
        this.constraints.add(f);
    }
}
