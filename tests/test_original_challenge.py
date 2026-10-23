"""Requirements-oriented checks for the course-map reconstruction.

These verify this new simulator, not the user's original 2025 hardware run.
"""

import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intralogistics_sim.geometry import World, distance, wrap_angle
from intralogistics_sim.original_challenge import (
    OriginalChallenge, TrackPIController, graph_shortest_path, load_map,
    simulate_original_challenge,
)
from intralogistics_sim.robot import DifferentialDrive, EncoderModel


class OriginalGraphTests(unittest.TestCase):
    def setUp(self):
        self.config = load_map()

    def test_costed_directed_storage_route_and_exit(self):
        inbound, inbound_cost = graph_shortest_path(self.config["graph"], "a", "i")
        outbound, outbound_cost = graph_shortest_path(self.config["graph"], "i", "n")
        self.assertEqual((inbound, inbound_cost), (["a", "f", "i"], 10))
        self.assertEqual((outbound, outbound_cost), (["i", "h", "j", "n"], 18))

    def test_direction_is_not_silently_reversed(self):
        with self.assertRaises(ValueError):
            graph_shortest_path(self.config["graph"], "o", "a")

    def test_line_geometry_covers_every_directed_graph_edge(self):
        graph_edges = {(a, b) for a, neighbors in self.config["graph"].items() for b in neighbors}
        geometry_edges = {(edge["from"], edge["to"]) for edge in self.config["warehouse_edges"]}
        self.assertEqual(graph_edges, geometry_edges)
        nodes = {node["id"]: [node["x"], node["y"]] for node in self.config["nodes"]}
        for edge in self.config["warehouse_edges"]:
            self.assertEqual(edge["points"][0], nodes[edge["from"]])
            self.assertEqual(edge["points"][-1], nodes[edge["to"]])

    def test_invalid_storage_is_rejected_before_simulation(self):
        with self.assertRaises(ValueError):
            OriginalChallenge(storage="a")


class OriginalControllerTests(unittest.TestCase):
    def test_closed_loop_corrects_initial_line_offset(self):
        robot = DifferentialDrive([1.0, 1.08, 0], EncoderModel(), wheelbase=0.16)
        robot.radius = 0.1
        controller = TrackPIController([[1, 1], [4, 1]], speed=0.28)
        world = World(6, 4, [])
        lateral_errors = []
        for _ in range(180):
            linear, angular = controller.command(robot.truth, 0.05)
            robot.step(linear, angular, 0.05, world)
            lateral_errors.append(abs(robot.truth[1] - 1.0))
        self.assertGreater(lateral_errors[0], 0.07)
        self.assertLess(lateral_errors[-1], 0.02)
        self.assertGreater(robot.truth[0], 3.0)
        self.assertEqual(robot.collisions, 0)


class OriginalChallengeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = simulate_original_challenge(storage="i", seed=42)

    def test_eight_original_tasks_complete_in_order(self):
        self.assertEqual([event["task"] for event in self.replay["events"] if event["kind"] == "task_complete"], list(range(1, 9)))
        self.assertTrue(self.replay["metrics"]["completed"])
        self.assertEqual(self.replay["metrics"]["collisions"], 0)

    def test_barcode_reject_then_match_then_load(self):
        checks = self.replay["barcode_checks"]
        self.assertEqual([(entry["id"], entry["matched"]) for entry in checks], [("EC-001", False), ("EC-002", True)])
        events = self.replay["events"]
        lift = next(event for event in events if event["kind"] == "fork_lift")
        match = next(event for event in events if event["kind"] == "barcode_match")
        self.assertGreater(lift["t"], match["t"])

    def test_worker_message_controls_actual_packing_choice(self):
        messages = self.replay["synthetic_messages"]
        self.assertEqual([message["topic"] for message in messages], ["sim/robot/arrival", "sim/worker/packaging"])
        requested = messages[-1]["payload"]
        selected = next(event for event in self.replay["events"] if event["kind"] == "shape_match")
        self.assertEqual(requested, selected["message"])
        self.assertGreater(selected["t"], messages[-1]["t"])

    def test_task6_waits_full_ten_seconds_and_rejoins_without_collision(self):
        start = next(event for event in self.replay["events"] if event["kind"] == "obstacle_detected")
        bypass = next(event for event in self.replay["events"] if event["kind"] == "bypass_start")
        self.assertAlmostEqual(bypass["t"] - start["t"], 10.0, places=5)
        self.assertLess(start["front_range"], 0.5)
        waiting = [frame for frame in self.replay["frames"] if start["t"] < frame["t"] < bypass["t"]]
        self.assertTrue(waiting)
        self.assertTrue(all(frame["command"]["linear"] == frame["command"]["angular"] == 0 for frame in waiting))
        self.assertLessEqual(self.replay["metrics"]["task6_max_side_gap"], 0.30)
        self.assertGreater(self.replay["metrics"]["task6_min_side_gap"], 0.10)

    def test_unmarked_return_waits_then_uses_encoder_pose(self):
        events = self.replay["events"]
        anchor = next(event for event in events if event["kind"] == "odometry_anchor")
        obstacle = next(event for event in events if event["kind"] == "return_obstacle")
        cleared = next(event for event in events if event["kind"] == "return_obstacle_cleared")
        self.assertLess(anchor["t"], obstacle["t"])
        self.assertAlmostEqual(cleared["t"] - obstacle["t"], 4.0, places=5)
        waiting = [frame for frame in self.replay["frames"] if obstacle["t"] < frame["t"] < cleared["t"]]
        self.assertTrue(all(frame["command"]["linear"] == 0 for frame in waiting))
        self.assertLess(self.replay["metrics"]["charging_error"], 0.06)
        self.assertGreater(self.replay["metrics"]["final_position_error"], 0.0001)

    def test_qr_follower_moves_by_measured_bearing_and_range(self):
        frames = [frame for frame in self.replay["frames"] if frame["worker_qr"]]
        self.assertGreater(len(frames), 40)
        self.assertGreater(distance(frames[0]["truth"], frames[-1]["truth"]), 3.0)
        self.assertGreater(distance(frames[0]["worker_qr"], frames[-1]["worker_qr"]), 3.0)
        self.assertLess(distance(frames[-1]["truth"], frames[-1]["worker_qr"]), 0.35)

    def test_all_allowed_storage_destinations_are_reachable(self):
        for node in ("g", "k", "l"):
            run = simulate_original_challenge(storage=node, seed=42)
            self.assertTrue(run["metrics"]["completed"])
            self.assertEqual(run["metrics"]["collisions"], 0)
            self.assertIn(node, run["warehouse_plan"]["nodes"])
            self.assertEqual(run["warehouse_plan"]["nodes"][-1], "n")

    def test_missing_barcode_match_fails_instead_of_inventing_success(self):
        config = load_map()
        config["requested_barcode"] = "MISSING-EC"
        simulation = OriginalChallenge(config)
        simulation.task_1()
        with self.assertRaises(RuntimeError):
            simulation.task_2()

    def test_unsupported_packaging_message_is_rejected(self):
        simulation = OriginalChallenge()
        simulation.packaging_shape = "Unsupported"
        simulation.robot.truth = [2, 2.9, -math.pi / 2]
        with self.assertRaises(RuntimeError):
            simulation.task_7()

    def test_same_seed_reproduces_numeric_frames(self):
        repeated = simulate_original_challenge(storage="i", seed=42)
        self.assertEqual(self.replay["frames"], repeated["frames"])
        self.assertEqual(self.replay["events"], repeated["events"])

    def test_playback_has_unique_times_and_continuous_physical_poses(self):
        frames = self.replay["frames"]
        boundaries = []
        for before, after in zip(frames, frames[1:]):
            elapsed = after["t"] - before["t"]
            self.assertGreater(elapsed, 0, "Duplicate timestamps break continuous playback interpolation")
            self.assertLessEqual(elapsed, 0.150001)
            # Plant limits must hold across task boundaries as well as within tasks.
            self.assertLessEqual(distance(before["truth"], after["truth"]), 0.28 * elapsed + 0.000002)
            yaw_change = abs(wrap_angle(after["truth"][2] - before["truth"][2]))
            self.assertLessEqual(yaw_change, 1.9 * elapsed + 0.000002)
            # Encoder bias/noise cannot justify a pose reset or visible teleport.
            self.assertLessEqual(distance(before["odom"], after["odom"]), 0.28 * elapsed + 0.001)
            if before["task"] != after["task"]:
                boundaries.append((before["task"], after["task"]))
        self.assertEqual(boundaries, list(zip(range(1, 8), range(2, 9))))

    def test_blue_marker_aligns_navigation_without_resetting_recorded_odom(self):
        anchor = next(event for event in self.replay["events"] if event["kind"] == "odometry_anchor")
        before = max((frame for frame in self.replay["frames"] if frame["t"] < anchor["t"]), key=lambda frame: frame["t"])
        after = min((frame for frame in self.replay["frames"] if frame["t"] >= anchor["t"]), key=lambda frame: frame["t"])
        self.assertLess(distance(before["odom"], after["odom"]), 0.045)
        self.assertGreater(after["metrics"]["position_error"], 0.02)
        self.assertLess(abs(after["metrics"]["position_error"] - before["metrics"]["position_error"]), 0.005)
        self.assertLess(self.replay["metrics"]["charging_error"], 0.06)

    def test_all_states_tasks_and_events_have_complete_english_text(self):
        self.assertTrue(self.replay["title_en"].isascii())
        for frame in self.replay["frames"]:
            self.assertTrue(frame["state_en"].strip())
            self.assertTrue(frame["state_en"].isascii())
        for event in self.replay["events"]:
            self.assertTrue(event["message_en"].strip())
            self.assertTrue(event["message_en"].isascii())
        for task in self.replay["tasks"]:
            self.assertTrue(task["label_en"].isascii())


if __name__ == "__main__":
    unittest.main()
