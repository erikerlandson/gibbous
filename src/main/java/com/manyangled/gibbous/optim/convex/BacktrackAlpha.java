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

public class BacktrackAlpha implements OptimizationData {
    public final double alpha;
    public BacktrackAlpha(double a) {
        if ((a <= 0.0) || (a >= 0.5))
            throw new IllegalArgumentException("alpha must be on (0,1/2)");
        this.alpha = a;
    }
}
