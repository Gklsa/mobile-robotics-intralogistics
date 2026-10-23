"""Behavioral checks for geometry, estimation, mapping, planning and task execution."""

import math
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intralogistics_sim.geometry import World, Rect, distance, warehouse_world
from intralogistics_sim.robot import DifferentialDrive, EncoderModel, integrate_pose
from intralogistics_sim.mapping import OccupancyGrid, bresenham
from intralogistics_sim.planning import dijkstra
from intralogistics_sim.scenarios import (simulate_mapping, simulate_challenge,
                                         simulate_bug2, simulate_line_following)


class GeometryTests(unittest.TestCase):
    def test_ray_obstacle_wall_and_no_return(self):
        world = World(10, 7, [Rect(4, 2, 1, 2)])
        self.assertEqual(world.raycast(1, 3, 0, 10), (3.0, True))
        self.assertEqual(world.raycast(1, 3, math.pi / 2, 10), (4.0, True))
        self.assertEqual(world.raycast(1, 3, 0, 2), (2, False))

    def test_circle_collision_and_clearance(self):
        world = World(10, 7, [Rect(4, 2, 1, 2)])
        self.assertTrue(world.collides(3.9, 3, 0.18))
        self.assertFalse(world.collides(3.7, 3, 0.18))
        self.assertFalse(world.segment_clear((3, 3), (6, 3)))
        self.assertTrue(world.segment_clear((3, 1), (6, 1)))


class EstimationTests(unittest.TestCase):
    def test_straight_and_exact_quarter_circle(self):
        self.assertEqual(integrate_pose([0, 0, 0], 1, 1, 0.4), [1, 0, 0])
        pose = integrate_pose([0, 0, 0], 0, math.pi / 2, 1)
        self.assertAlmostEqual(pose[0], 0.5)
        self.assertAlmostEqual(pose[1], 0.5)
        self.assertAlmostEqual(pose[2], math.pi / 2)

    def test_exact_encoders_follow_truth_without_bias(self):
        world = World(10, 10, [])
        robot = DifferentialDrive([2, 2, 0], EncoderModel(resolution=1e-8))
        for _ in range(100):
            robot.step(0.4, 0.2, 0.05, world)
        self.assertLess(distance(robot.truth, robot.odom), 1e-6)
        self.assertLess(abs(robot.truth[2] - robot.odom[2]), 1e-6)

    def test_collision_does_not_teleport_robot(self):
        world = World(4, 4, [Rect(2, 1, 1, 2)])
        robot = DifferentialDrive([1.7, 2, 0])
        self.assertFalse(robot.step(1, 0, 0.5, world))
        self.assertEqual(robot.truth, [1.7, 2, 0])
        self.assertEqual(robot.collisions, 1)


class MappingPlanningTests(unittest.TestCase):
    def test_inverse_sensor_free_and_hit_and_unknown(self):
        grid = OccupancyGrid(World(5, 3, []), 0.2)
        updates = grid.update([1, 1, 0], [{"angle": 0, "range": 1, "hit": True}])
        self.assertTrue(updates)
        self.assertEqual(grid.values[5 * grid.width + 5], 0)
        self.assertEqual(grid.values[5 * grid.width + 10], 100)
        self.assertEqual(grid.values[0], -1)
        self.assertEqual(bresenham(0, 0, 2, 0), [(0, 0), (1, 0), (2, 0)])

    def test_no_return_does_not_invent_occupied_endpoint(self):
        grid = OccupancyGrid(World(5, 3, []), 0.2)
        grid.update([1, 1, 0], [{"angle": 0, "range": 1, "hit": False}])
        self.assertNotIn(100, grid.values)

    def test_dijkstra_detour_reaches_endpoints_and_preserves_clearance(self):
        world = World(10, 7, [Rect(4, 1.5, 1, 4)])
        start, goal = [1, 3.5], [9, 3.5]
        route = dijkstra(world, start, goal)
        self.assertEqual(route[0], start)
        self.assertEqual(route[-1], goal)
        self.assertGreater(sum(distance(a, b) for a, b in zip(route, route[1:])), 8)
        self.assertTrue(all(world.segment_clear(a, b, 0.28) for a, b in zip(route, route[1:])))

    def test_unreachable_goal_is_rejected(self):
        world = World(4, 4, [Rect(1.8, 0, 0.4, 4)])
        with self.assertRaises(ValueError):
            dijkstra(world, [1, 2], [3, 2])


class ScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = simulate_mapping("low")
        cls.noisy = simulate_mapping("high")
        cls.challenge = simulate_challenge()
        cls.bug = simulate_bug2()
        cls.line = simulate_line_following()

    def test_mapping_both_complete_same_real_route_without_collisions(self):
        for run in (self.clean, self.noisy):
            self.assertTrue(run["metrics"]["completed"])
            self.assertEqual(run["metrics"]["collisions"], 0)
            self.assertGreater(run["metrics"]["distance"], 25)
            self.assertGreater(run["metrics"]["known_fraction"], 0.5)
        self.assertEqual([f["truth"] for f in self.clean["frames"]],
                         [f["truth"] for f in self.noisy["frames"]])

    def test_encoder_bias_increases_actual_error_and_distorts_map(self):
        self.assertGreater(self.noisy["metrics"]["position_rmse"],
                           self.clean["metrics"]["position_rmse"] * 10)
        self.assertGreater(self.noisy["metrics"]["final_position_error"],
                           self.clean["metrics"]["final_position_error"] + 0.1)
        self.assertNotEqual(self.clean["frames"][-1]["odom"], self.noisy["frames"][-1]["odom"])
        clean_map, noisy_map = {}, {}
        for run, grid in ((self.clean, clean_map), (self.noisy, noisy_map)):
            for frame in run["frames"]:
                grid.update(frame["updates"])
        self.assertGreater(sum(clean_map.get(cell, -1) != value for cell, value in noisy_map.items()), 50)

    def test_task_order_and_synthetic_message_causality(self):
        run = self.challenge
        completed = [event["task"] for event in run["events"] if event["kind"] == "complete"]
        self.assertEqual(completed, list(range(1, 9)))
        self.assertTrue(run["metrics"]["completed"])
        self.assertEqual(run["metrics"]["collisions"], 0)
        self.assertLess(distance(run["frames"][-1]["truth"], [1, 1]), 0.08)
        self.assertEqual(len(run["synthetic_messages"]), 4)
        self.assertGreater(run["synthetic_messages"][0]["t"], 1.9)

    def test_bug2_boundary_then_closer_m_line_leave(self):
        run = self.bug
        self.assertTrue(run["metrics"]["completed"])
        self.assertEqual(run["metrics"]["collisions"], 0)
        self.assertEqual(run["bug2"]["transitions"], ["绕障边界跟随", "返回 M 线", "到达终点"])
        self.assertLess(run["bug2"]["leave_distance_to_goal"], run["bug2"]["hit_distance_to_goal"])
        self.assertGreater(run["metrics"]["distance"], 8)
        self.assertLess(min(frame["truth"][1] for frame in run["frames"]), 2.0)

    def test_pi_rejects_bias_better_under_identical_conditions(self):
        p, pi = self.line["runs"]
        self.assertEqual(p["samples"][0]["error"], pi["samples"][0]["error"])
        self.assertGreater(p["metrics"]["steady_rms_error"], 0.04)
        self.assertLess(pi["metrics"]["steady_rms_error"], p["metrics"]["steady_rms_error"] / 5)
        self.assertLess(abs(pi["metrics"]["final_error"]), 0.01)
        self.assertGreater(pi["samples"][-1]["x"], 11)

    def test_replay_changes_have_valid_grid_indices_and_observed_values(self):
        for run in (self.clean, self.noisy, self.challenge, self.bug):
            size = run["grid"]["width"] * run["grid"]["height"]
            for frame in run["frames"]:
                self.assertTrue(all(0 <= cell < size and value in (0, 100) for cell, value in frame["updates"]))
                self.assertEqual(len(frame["rays"]), 7)


if __name__ == "__main__":
    unittest.main()
