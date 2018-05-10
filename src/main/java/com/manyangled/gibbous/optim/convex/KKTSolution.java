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

public class KKTSolution {
    public final RealVector xDelta;
    public final RealVector nuPlus;
    public final double lambdaSquared;

    // Used for Newton's method with equality constraints (alg 10.2)
    KKTSolution(final RealVector xd, final RealVector nup) {
        this.xDelta = xd;
        this.nuPlus = nup;
        this.lambdaSquared = 0.0;
    }

    // Used for Newton's method without constraints (alg 9.5)
    KKTSolution(final RealVector xd, final double lsq) {
        this.xDelta = xd;
        this.lambdaSquared = lsq;
        this.nuPlus = null;
    }
}
