# gibbous
Convex optimization built on Apache Commons Math

Implementation of the Barrier Method from §11.3 of _Convex Optimization_, Boyd and Vandenberghe, Cambridge University Press, 2004

### Documentation
Full API javadoc is available at: https://erikerlandson.github.io/gibbous/java/api/

Some examples are included below.

### How to include `gibbous` in your project
`gibbous` is a java project, and so can be used either with java or scala.

Versions prior to `gibbous 0.3.0` were published to bintray,
which is no longer operative. It is now available through oss.sonatype.org

`gibbous 0.3.0` is equivalent to `0.2.x`, with the exception of now including
`"org.apache.commons" % "commons-math3" % "3.6.1"` as an explicit dependency.

```scala
libraryDependencies ++= "com.manyangled" % "gibbous" % "0.3.0"
```

### Examples

##### Minimize a convex function under constraints
```java
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;

import com.manyangled.gibbous.optim.convex.*;

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

##### Using a feasible point solver to get a feasible initial guess
```java
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;

import com.manyangled.gibbous.optim.convex.*;

// create a convex objective function
QuadraticFunction q = new QuadraticFunction(
    new double[][] { { 1.0, 0.0 }, { 0.0, 1.0 } },
    new double[] { 0.0, 0.0 },
    0.0);

// Declare constraints separately to use for solving a feasible point
LinearInequalityConstraint ineqc = new LinearInequalityConstraint(
    new double[][] { { -1.0, 0.0 } }, // constraint x > 1,
    new double[] { -1.0 });
LinearEqualityConstraint eqc = new LinearEqualityConstraint(
    new double[][] { { 0.0, 1.0 } },  // constraint y = 1,
    new double[] { 1.0 });

// solve for a feasible point that satisfies the constraints
PointValuePair fpvp = ConvexOptimizer.feasiblePoint(ineqc, eqc);
// if not < 0, there is no feasible point
assert fpvp.getSecond() < 0.0;
double[] ig = fpvp.getFirst();

// optimize function q with the same contraints, using the feasible point
// for the initial guess
BarrierOptimizer barrier = new BarrierOptimizer();
PointValuePair pvp = barrier.optimize(
    new ObjectiveFunction(q),
    ineqc,
    eqc,
    new InitialGuess(ig));

double[] xmin = pvp.getFirst();  // { 1.0, 1.0 }
double vmin = pvp.getSecond();   // 1.0
```
