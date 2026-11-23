"""Joint configurations have J entries; spatial twists always have six."""

import unittest

import kincpp
import numpy as np


def _model(joints):
    axes = np.array(
        [
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ],
        dtype=float,
    )[:joints]
    origins = np.array(
        [
            [0, 0, 0.1],
            [0, 0, 0.2],
            [0.3, 0, 0.2],
            [0.6, 0, 0.2],
            [0.6, 0, 0.3],
            [0.6, 0, 0.4],
            [0.7, 0, 0.4],
        ],
        dtype=float,
    )[:joints]
    screws = np.concatenate([axes, -np.cross(axes, origins)], axis=1).T
    home = np.eye(4)
    home[:3, 3] = [0.75, 0.05, 0.4]
    return home, screws, np.full(joints, -2.0), np.full(joints, 2.0)


class JointDimensionsTest(unittest.TestCase):
    def test_parallel_flags_match_exact_fk_targets(self):
        solver = kincpp.KinematicsSolver(*_model(7))
        desired = np.zeros(7)
        target = solver.forward_kinematics(desired)
        parameters = kincpp.IKParams()
        parameters.batched_common_params.desired_ee_tfs = [target] * 128
        parameters.batched_common_params.q_guesses = np.tile(desired, (128, 1))
        parameters.batched_common_params.iters = 20
        # Exact seeds finish together, stressing adjacent convergence flags
        # written by different OpenMP workers. No optimization can fail here.
        for repeat in range(100):
            flags, solutions = solver.batched_inverse_kinematics(parameters)
            with self.subTest(repeat=repeat):
                self.assertTrue(all(flags))
                np.testing.assert_array_equal(
                    solutions, np.tile(desired, (128, 1))
                )

    def test_six_and_seven_joint_single_and_batched_solve(self):
        for joints in (6, 7):
            with self.subTest(joints=joints):
                solver = kincpp.KinematicsSolver(*_model(joints))
                desired = np.linspace(-0.12, 0.15, joints)
                target = solver.forward_kinematics(desired)
                parameters = kincpp.IKParams()
                parameters.newton_params.position_tolerance = 1e-7
                parameters.newton_params.orientation_tolerance = 1e-7
                parameters.newton_params.project_to_joint_limits = True
                parameters.common_params.iters = 80
                parameters.common_params.desired_ee_tf = target
                parameters.common_params.q_guess = desired + 0.02
                success, solution = solver.inverse_kinematics(parameters)
                self.assertTrue(success)
                self.assertEqual(solution.shape, (joints,))
                np.testing.assert_allclose(
                    solver.forward_kinematics(solution), target, atol=1e-6
                )
                self.assertTrue(np.all(np.abs(solution) <= 2))
                parameters.batched_common_params.iters = 80
                parameters.batched_common_params.desired_ee_tfs = [target] * 2
                parameters.batched_common_params.q_guesses = np.stack(
                    [desired + 0.02, desired - 0.01]
                )
                flags, solutions = solver.batched_inverse_kinematics(
                    parameters
                )
                self.assertTrue(all(flags))
                self.assertEqual(solutions.shape, (2, joints))
                for achieved in solver.batched_forward_kinematics(solutions):
                    np.testing.assert_allclose(achieved, target, atol=1e-6)

    def test_dimension_errors_raise_before_native_linear_algebra(self):
        solver = kincpp.KinematicsSolver(*_model(7))
        with self.assertRaises(ValueError):
            solver.forward_kinematics(np.zeros(6))
        with self.assertRaises(ValueError):
            solver.batched_forward_kinematics(np.zeros((2, 6)))
        parameters = kincpp.IKParams()
        parameters.common_params.q_guess = np.zeros(6)
        parameters.common_params.desired_ee_tf = np.eye(4)
        with self.assertRaises(ValueError):
            solver.inverse_kinematics(parameters)
        parameters.batched_common_params.q_guesses = np.zeros((2, 7))
        parameters.batched_common_params.desired_ee_tfs = [np.eye(4)]
        with self.assertRaises(ValueError):
            solver.batched_inverse_kinematics(parameters)

    def test_invalid_model_dimensions_are_rejected(self):
        home, screws, low, high = _model(7)
        for model in (
            (np.eye(3), screws, low, high),
            (home, screws[:5], low, high),
            (home, screws, low[:6], high),
            (home, screws, high, low),
        ):
            with self.subTest(shapes=[value.shape for value in model]):
                with self.assertRaises(ValueError):
                    kincpp.KinematicsSolver(*model)


if __name__ == "__main__":
    unittest.main()
