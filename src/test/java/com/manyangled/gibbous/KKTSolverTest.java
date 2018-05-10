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

package com.manyangled.gibbous;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertArrayEquals;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;

import com.manyangled.gibbous.optim.convex.KKTSolver;
import com.manyangled.gibbous.optim.convex.KKTSolution;
import com.manyangled.gibbous.optim.convex.SchurKKTSolver;

public class KKTSolverTest {
    private void testWithConstraints(
        KKTSolver kkts,
        double[][] Hdata, double[][] Adata, double[] gdata, double[] hdata) {
        RealMatrix H = new Array2DRowRealMatrix(Hdata);
        RealMatrix A = new Array2DRowRealMatrix(Adata);
        RealVector g = new ArrayRealVector(gdata);
        RealVector h = new ArrayRealVector(hdata);
        RealMatrix AT = A.transpose();
        KKTSolution sol = kkts.solve(H, A, AT, g, h);
        int n = Hdata.length + Adata.length;
        int Hn = Hdata.length;
        double[][] Tdata = new double[n][n];
        for (int j = 0; j < Hn; ++j)
            for (int k = 0; k < Hn; ++k)
                Tdata[j][k] = Hdata[j][k];
        for (int j = 0; j < Adata.length; ++j)
            for (int k = 0; k < Adata[0].length; ++k) {
                Tdata[Hn + j][k] = Adata[j][k];
                Tdata[k][Hn + j] = Adata[j][k];
            }
        RealMatrix T = new Array2DRowRealMatrix(Tdata);
        RealVector t = sol.xDelta.append(sol.nuPlus);
        RealVector Txt = T.operate(t);
        RealVector target = g.append(h).mapMultiply(-1.0);
        assertArrayEquals(target.toArray(), Txt.toArray(), 1e-9);
    }

    @Test
    public void testSchurConstrained1() {
        double[][] H = { { 2.0, 1.0 }, { 1.0, 2.0 } };
        double[][] A = { { 1.0, 1.0 } };
        double[] g = { 1.0, 2.0 };
        double[] h = { 3.0 };
        testWithConstraints(new SchurKKTSolver(), H, A, g, h);
    }

    @Test
    public void testSchurConstrained2() {
        double[][] H = { { 3.0, 2.0, 1.0 },
                         { 2.0, 3.0, 2.0 },
                         { 1.0, 2.0, 3.0 } };
        double[][] A = { { 1.0, 2.0, 3.0 },
                         { 6.0, 5.0, 4.0 } };
        double[] g = { 1.0, 4.0, 9.0 };
        double[] h = { 3.0, 7.0 };
        testWithConstraints(new SchurKKTSolver(), H, A, g, h);
    }
}
