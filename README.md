# gibbous
Convex optimization built on Apache Commons Math

Implementation of the Barrier Method from §11.3 of _Convex Optimization_, Boyd and Vandenberghe, Cambridge University Press, 2004

### Documentation
Full API javadoc is available at: https://erikerlandson.github.io/gibbous/java/api/

### Examples

```java
// create a convex objective function
QuadraticFunction q = new QuadraticFunction(
    new double[][] { { 1.0, 0.0 }, { 0.0, 1.0 } },
    new double[] { 0.0, 0.0 },
    0.0);

// optimize function q with an inequality constraint and an equality constraint,
// using the barrier method
BarrierOptimizer barrier = new BarrierOptimizer();
PointValuePair pvp = barrier.optimize(
    new ObjectiveFunction(q),
    new LinearInequalityConstraint(
        new double[][] { { -1.0, 0.0 } }, // constraint x > 1,
        new double[] { -1.0 }),
    new LinearEqualityConstraint(
        new double[][] { { 0.0, 1.0 } },  // constraint y = 1,
        new double[] { 1.0 }),
    new InitialGuess(new double[] { 10.0, 10.0 }));

double[] xmin = pvp.getFirst();  // { 1.0, 1.0 }
double vmin = pvp.getSecond();   // 1.0
```
