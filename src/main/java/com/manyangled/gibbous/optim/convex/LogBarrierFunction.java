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
    private final boolean zh;

    LogBarrierFunction(int nd, Collection<ConvexFunction> cf) {
        if (nd < 1) {
            throw new IllegalArgumentException("domain dimension nd must be positive");
        }
        int m = cf.size();
        f = new ConvexFunction[m];
        Iterator e = cf.iterator();
        int j = 0;
        boolean allZH = true;
        while (e.hasNext()) {
            f[j] = (ConvexFunction)(e.next());
            if (f[j].dim() != nd) {
                throw new DimensionMismatchException(f[j].dim(), nd);
            }
            if (!f[j].zeroHessian()) allZH = false;
            j += 1;
        }
        n = nd;
        gw = new double[n];
        hw = new double[n][n];
        zh = allZH;
    }

    @Override
    public int dim() {
        return n;
    }

    @Override
    public boolean zeroHessian() { return zh; }

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
            if (fi.zeroHessian()) continue;
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
