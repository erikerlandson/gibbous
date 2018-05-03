package com.manyangled.gibbous.optim.convex;

import java.lang.Math;
import java.util.Collection;
import java.util.Iterator;
import java.util.Arrays;
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.exception.DimensionMismatchException;

public class LogBarrierFunction extends ConvexFunction
    implements OptimizationData {

    private final ConvexFunction[] f;
    private final double[] gw;
    private final double[][] hw;
    private final int n;

    LogBarrierFunction(Collection<ConvexFunction> cf) {
        int m = cf.size();
        f = new ConvexFunction[m];
        Iterator e = cf.iterator();
        int j = 0;
        while (e.hasNext()) {
            f[j] = (ConvexFunction)(e.next());
            if (f[j].dim() != f[0].dim()) {
                throw new DimensionMismatchException(f[j].dim(), f[0].dim());
            }
            j += 1;
        }
        n = (m > 0) ? f[0].dim() : 0;
        gw = new double[n];
        hw = new double[n][n];
    }

    @Override
    public int dim() {
        return n;
    }

    @Override
    public double value(final double[] x) {
        double v = 0.0;
        for (ConvexFunction fi: f) {
            double t = fi.value(x);
            if (t >= 0.0) {
                return Double.POSITIVE_INFINITY;
            }
            v += Math.log(-t);
        }
        return -v;
    }

    @Override
    public void fillGradient(final double[] x, double[] g) {
        if (x.length != n) {
            throw new DimensionMismatchException(g.length, n);
        }
        if (g.length != n) {
            throw new DimensionMismatchException(g.length, n);
        }
        Arrays.fill(g, 0.0);
        for (ConvexFunction fi: f) {
            double vi = fi.value(x);
            fi.fillGradient(x, gw);
            for (int j = 0; j < n; ++j) {
                g[j] -= gw[j]/(vi);
            }
        }
    }

    @Override
    public void fillHessian(final double[] x, double[][] h) {
        if (x.length != n) {
            throw new DimensionMismatchException(x.length, n);
        }
        if (h.length != n) {
            throw new DimensionMismatchException(h.length, n);
        }
        for (int j = 0; j < n; ++j) {
            if (h[j].length != n) {
                throw new DimensionMismatchException(h[j].length, n);
            }
            Arrays.fill(h[j], 0.0);
        }
        for (ConvexFunction fi: f) {
            double vi = fi.value(x);
            fi.fillGradient(x, gw);
            fi.fillHessian(x, hw);
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                    h[j][k] += (gw[j]*gw[k]/(vi*vi)) - hw[j][k]/vi;
                }
            }
        }
    }
}
