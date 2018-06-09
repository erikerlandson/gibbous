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
 * A collection of {@link OptimizationData} arguments that are passed to an inner optimizer.
 * <p>
 * For example, the {@link BarrierOptimizer} operates by iterative calls to the
 * {@link NewtonOptimizer}. A user may pass configuration arguments directly to the
 * {@link NewtonOptimizer} by putting them in a {@link InnerOptimizationData} argument
 * while invoking {@link BarrierOptimizer#optimize(OptimizationData... optData)}
 */
public class InnerOptimizationData implements OptimizationData {
    public final ArrayList<OptimizationData> optData =
        new ArrayList<OptimizationData>();

    /**
     * Create an InnerOptimizationData payload from a Collection of OptimizationData
     * @param optData the Collection of OptimizationData arguments
     */
    public InnerOptimizationData(Collection<OptimizationData> optData) {
        this.optData.addAll(optData);
    }

    /**
     * Create an InnerOptimizationData payload from an argument list (or array)
     * of OptimizationData
     * @param optData the list of OptimizationData arguments
     */
    public InnerOptimizationData(OptimizationData... optData) {
        for (OptimizationData data: optData)
            this.optData.add(data);
    }
}
