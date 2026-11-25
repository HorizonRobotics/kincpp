# TODO: correct the angle scalar in MatrixLog6

Status: recorded; the numerical implementation is unchanged.

In `src/utils.cpp`, `MatrixLog6` computes `omgmat = MatrixLog3(R)` and then
uses `omgmat.norm()` as the rotation angle. Eigen takes the matrix's
Frobenius norm, which is `sqrt(2)` times the rotation-vector norm.

The intended correction is:

```cpp
double theta = SO3ToVec(omgmat).norm();
```

The wrong scalar biases the linear SE(3)-log coefficient
`1/theta^2 - cot(theta/2)/(2*theta)`. The angular skew matrix is copied into
the result without this scaling, so Newton's angular-vector stopping test
still uses the specified tolerance. It does not turn 0.01 rad into
`0.01 / sqrt(2)` rad. Native zero-update probes accept 0.00999 rad and
reject 0.01001 rad under a 0.01-rad orientation tolerance.

Before fixing this, add mixed rotation/translation regressions verifying
`MatrixExp6(MatrixLog6(T))` reproduces `T`, including translations
perpendicular to the rotation axis, several axes, small angles, and
near-pi rotations. Keep this change separate from the Cartesian TCP
convergence correction in commit `bdff69f` and record its own solver
comparison condition.

The 2026-11-25 native audit used commit
`bdff69f1ef870d07c3eeb3ae3675d8d557823f90`. The linear-log perturbation is
fourth order near zero: approximately
`angle^4 * norm(perpendicular_translation) / 720`.
For 0.1 m perpendicular translation, it is approximately 1.39e-12 m at
0.01 rad, 0.149 mm at 1 rad, and 3.046 mm at 2 rad. These are residual
perturbations, not measured TCP tracking errors.

An isolated one-line correction, rebuilt with the same compiler/options
and SciPy refinement disabled, left the ten-demo offline IK outcome at
8/10 complete solves. The same two cases were refused at the same
waypoints; maximum returned arm-joint component change on common accepted
prefixes was 2.58614e-7 rad. No policy benchmark effect has been measured.

[Native paired audit and reproduction scripts](http://gail7:3000/home/horizon/tmp/geniesim3_matrixlog6_ablation_2026-11-25/index.html)
retain source/binary/compiler, processor and demo-manifest provenance.
