"""Newton convergence must use actual TCP displacement, not spatial v."""

import unittest

import kincpp
import numpy as np


class PositionConvergenceTest(unittest.TestCase):
    def make_problem(self, origin=0.0, lower=-3.0, upper=3.0):
        home = np.eye(4)
        home[0, 3] = origin + 1.0
        # A revolute Z joint at (origin, 0, 0), with a one-meter TCP lever.
        screws = np.array([[0.0], [0.0], [1.0], [0.0], [-origin], [0.0]])
        solver = kincpp.KinematicsSolver(
            home, screws, np.array([lower]), np.array([upper])
        )
        params = kincpp.IKParams()
        params.solver_type = kincpp.NEWTON
        params.newton_params.position_tolerance = 0.001
        params.newton_params.orientation_tolerance = 0.01
        params.newton_params.project_to_joint_limits = True
        params.common_params.q_guess = np.array([0.0])
        params.common_params.iters = 100
        params.common_params.desired_ee_tf = solver.forward_kinematics(
            np.array([0.005])
        )
        return solver, params

    def test_small_rotation_still_requires_tcp_position_refinement(self):
        for origin in (0.0, 1.0, 100.0):
            with self.subTest(origin=origin):
                solver, params = self.make_problem(origin)
                success, solution = solver.inverse_kinematics(params)
                actual = solver.forward_kinematics(solution)
                self.assertTrue(success)
                self.assertLess(
                    np.linalg.norm(
                        actual[:3, 3]
                        - params.common_params.desired_ee_tf[:3, 3]
                    ),
                    0.001,
                )

    def test_acceptable_pose_is_accepted_without_unnecessary_motion(self):
        solver, params = self.make_problem()
        target = params.common_params.desired_ee_tf.copy()
        target[:3, 3] = [1.0, 0.0, 0.0]
        params.common_params.desired_ee_tf = target
        success, solution = solver.inverse_kinematics(params)
        self.assertTrue(success)
        np.testing.assert_array_equal(solution, [0.0])

    def test_final_limit_projection_cannot_retain_old_success_flag(self):
        solver, params = self.make_problem(lower=-0.001, upper=0.001)
        params.common_params.q_guess = np.array([0.005])
        params.newton_params.project_to_joint_limits = False
        success, solution = solver.inverse_kinematics(params)
        self.assertFalse(success)
        self.assertLessEqual(solution[0], 0.001)


if __name__ == "__main__":
    unittest.main()
