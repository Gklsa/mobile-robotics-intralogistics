"""New simulation of the *original* eight-task course Challenge.

The line network is redrawn as editable geometric data from the supplied course
diagram.  The 2025 physical course success is distinct from this 2026 rebuild.
No instructor implementation, PDF or image is bundled.  Metric scale is assumed.
"""

import argparse
import heapq
import json
import math
from pathlib import Path
import random
import re

from .geometry import World, Rect, distance, wrap_angle
from .robot import DifferentialDrive, EncoderModel


ROOT = Path(__file__).resolve().parents[2]
TASKS = [
    {"id": 1, "label": "交通灯变绿后启动", "label_en": "Wait for green traffic light"},
    {"id": 2, "label": "条码匹配并提取货箱", "label_en": "Barcode-match container pickup"},
    {"id": 3, "label": "Dijkstra 入库与指定出口", "label_en": "Store container / exit at n"},
    {"id": 4, "label": "MQTT 到站与包装形状", "label_en": "Worker arrival / packaging message"},
    {"id": 5, "label": "跟随人工引导的 QR", "label_en": "Follow manually guided QR"},
    {"id": 6, "label": "等待 10 秒后近距绕障", "label_en": "Wait 10 s / bypass obstacle"},
    {"id": 7, "label": "按形状选择包装站并卸货", "label_en": "Shape-match packing station"},
    {"id": 8, "label": "无标线返回充电站", "label_en": "Return to charging station"},
]


STATE_ENGLISH = {
    "准备启动": "Ready to start",
    "八项原始 Challenge 任务完成": "All eight original Challenge tasks complete",
    "等待合成交通灯由红变绿": "Wait for simulated traffic light to turn green",
    "PI 沿黑线驶向收货区": "PI line following to goods receipt",
    "逐站比较条码": "Compare container barcodes at each station",
    "叉取已匹配 EC": "Pick up the matched container",
    "合成叉架提升": "Simulated fork lift",
    "返回黑线": "Rejoin the black line",
    "进入仓储线网": "Enter the warehouse line network",
    "合成叉架卸货": "Simulated fork unloading",
    "沿有向线网离开仓库，出口 n": "Follow the directed line network to exit n",
    "在蓝色标记处到站": "Stop at the blue arrival marker",
    "等待合成工人消息": "Wait for the simulated worker message",
    "离开交互站": "Leave the worker interaction station",
    "根据合成 QR 方位与距离闭环跟随": "Follow simulated QR bearing and range",
    "沿线行驶并检测前方障碍": "Follow the line and sense a frontal obstacle",
    "障碍未离开，等待 10 秒": "Wait 10 seconds for obstacle clearance",
    "近距绕障并重新找回黑线": "Pass close to the obstacle and reacquire the line",
    "返回黑线并驶向包装区": "Rejoin the black line toward the packing area",
    "沿黑线逐站识别包装形状": "Follow the line and inspect packing shapes",
    "交付至匹配的包装站": "Deliver to the matching packing station",
    "合成卸货完成": "Simulated unloading",
    "返回主线": "Rejoin the main line",
    "沿线返回，直至蓝色终止标记": "Follow the line to the blue end marker",
    "等待障碍离开，不绕行": "Stop until the obstacle clears; no bypass",
    "无标线编码器导航至充电站": "Encoder-only navigation to the charging station",
}


def english_state(state):
    """A complete English view of existing states, preserving Chinese source data."""
    match = re.fullmatch(r"任务 (\d+) · (.*)", state)
    prefix, detail = (f"Task {match[1]} - ", match[2]) if match else ("", state)
    if detail in STATE_ENGLISH:
        return prefix + STATE_ENGLISH[detail]
    for old, new in (("读取合成条码 ", "Read simulated barcode "),
                     ("Dijkstra 到货位 ", "Dijkstra route to storage "),
                     ("合成轮廓观察：", "Simulated contour observation: ")):
        if detail.startswith(old):
            return prefix + new + detail[len(old):]
    if detail.isascii():
        return prefix + detail
    raise ValueError(f"Untranslated simulator state: {state}")


def english_event(kind, message, task):
    """Translate event semantics rather than hiding them behind numeric task IDs."""
    fixed = {
        "green_light": "Simulated camera green-pixel fraction exceeded the threshold",
        "worker_container": "Worker handed over an empty container (simulated event)",
        "qr_follow_complete": "Manually guided QR reached its destination; not markerless person tracking",
        "obstacle_detected": "Front range sensor detected an obstacle; start the 10-second wait",
        "bypass_start": "Wait complete; begin geometric bypass with a 0.30 m maximum side-gap requirement",
        "line_end": "Blue marker ends line sensing; switch to encoder-only navigation",
        "odometry_anchor": "Known blue marker anchors the local navigation origin; recorded odometry remains continuous",
        "return_obstacle": "Obstacle on unmarked return route: stop and wait for it to leave",
        "return_obstacle_cleared": "Simulated return obstacle cleared; resume encoder navigation",
        "charging": "Charging station reached; simulation parked, no real electrical charging",
    }
    if kind in fixed:
        return fixed[kind]
    if kind == "task_start":
        return TASKS[task - 1]["label_en"]
    if kind == "task_complete":
        return f"Task {task} complete"
    if kind == "qr_task":
        return f"Simulated task QR payload: {task}; semantic sensor input, not image decoding"
    if kind == "barcode_request":
        return "Requested container barcode: " + message.split("：")[-1]
    if kind in ("barcode_match", "barcode_reject", "shape_match", "shape_reject"):
        description = {"barcode_match": "Barcode matched", "barcode_reject": "Barcode rejected",
                       "shape_match": "Packing shape matched", "shape_reject": "Packing shape rejected"}[kind]
        return f"{description}: {message}"
    if kind == "fork_lift":
        return "Container loaded: " + message.split()[-1]
    if kind == "storage_drop":
        return "Container stored at node " + message.split()[-1]
    if kind == "plan":
        return "Directed Dijkstra path: " + message.replace(" → ", " -> ")
    if message.isascii():
        return message
    raise ValueError(f"Untranslated simulator event: {kind}: {message}")


def load_map(path=None):
    return json.loads(Path(path or ROOT / "config" / "original_map.json").read_text(encoding="utf-8"))


def graph_shortest_path(graph, start, goal):
    """New directed Dijkstra implementation using abstract supplied edge costs."""
    frontier, distances, parents = [(0, start)], {start: 0}, {}
    while frontier:
        cost, node = heapq.heappop(frontier)
        if cost != distances[node]:
            continue
        if node == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parents[path[-1]])
            return list(reversed(path)), cost
        for neighbor, edge_cost in graph[node].items():
            if edge_cost < 0:
                raise ValueError("Dijkstra requires nonnegative edge costs")
            candidate = cost + edge_cost
            if candidate < distances.get(neighbor, math.inf):
                distances[neighbor], parents[neighbor] = candidate, node
                heapq.heappush(frontier, (candidate, neighbor))
    raise ValueError(f"No directed path from {start} to {goal}")


def polyline_length(points):
    return sum(distance(first, second) for first, second in zip(points, points[1:]))


def along_polyline(points, traveled):
    for first, second in zip(points, points[1:]):
        length = distance(first, second)
        if traveled <= length:
            fraction = 0 if length == 0 else max(0, traveled / length)
            return [first[0] + (second[0] - first[0]) * fraction,
                    first[1] + (second[1] - first[1]) * fraction]
        traveled -= length
    return list(points[-1])


def point_rectangle_distance(point, rectangle):
    dx = max(rectangle.x - point[0], 0, point[0] - rectangle.x - rectangle.w)
    dy = max(rectangle.y - point[1], 0, point[1] - rectangle.y - rectangle.h)
    return math.hypot(dx, dy)


class TrackPIController:
    """Synthetic line-feature PI controller with heading feedback.

    The virtual line sensor obtains cross-track offset and line heading from the
    selected stroke; it is a geometric sensor abstraction, not image processing.
    During the final unmarked leg the same steering law receives *odometry*.
    """
    def __init__(self, path, speed=0.28, tolerance=0.035):
        self.path = [list(point) for point in path]
        self.index = 1
        self.speed, self.tolerance = speed, tolerance
        self.integral = 0.0
        self.cross_track = 0.0

    def command(self, pose, dt):
        while self.index < len(self.path) and distance(pose, self.path[self.index]) < self.tolerance:
            self.index += 1
            self.integral = 0.0
        if self.done:
            return 0.0, 0.0
        start, target = self.path[self.index - 1], self.path[self.index]
        dx, dy = target[0] - start[0], target[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1e-8:
            self.index += 1
            return self.command(pose, dt)
        self.cross_track = (dx * (pose[1] - start[1]) - dy * (pose[0] - start[0])) / length
        self.integral = max(-0.15, min(0.15, self.integral + self.cross_track * dt))
        desired = math.atan2(target[1] - pose[1], target[0] - pose[0])
        heading_error = wrap_angle(desired - pose[2])
        angular = max(-1.9, min(1.9, 3.2 * heading_error - 1.2 * self.cross_track - 0.25 * self.integral))
        linear = min(self.speed, 1.6 * distance(pose, target)) * max(0, math.cos(heading_error)) ** 3
        return linear, angular

    @property
    def done(self):
        return self.index >= len(self.path)


class OriginalChallenge:
    """Deterministic task execution against simulated sensor and worker events."""
    dt = 0.05

    def __init__(self, config=None, storage="i", seed=42):
        self.config = config or load_map()
        self.storage = storage
        if storage not in self.config["storage_nodes"]:
            raise ValueError("Storage must be one of g / i / k / l")
        self.seed, self.rng = seed, random.Random(seed)
        self.world = World(self.config["width"], self.config["height"], [])
        # Mild encoder uncertainty is intentionally visible on the unmarked leg.
        noise = EncoderModel(1.0001, 0.9999, 1.0001, 0.000006, 0.0001)
        self.robot = DifferentialDrive(self.config["start"], noise, seed,
                                       wheelbase=self.config["robot"]["wheelbase"])
        self.robot.radius = self.config["robot"]["radius"]
        self.t, self.task, self.steps = 0.0, 1, 0
        self.loaded, self.worker_qr = False, None
        self.frames, self.events, self.messages, self.route = [], [], [], []
        self.active_obstacles = []
        self.packaging_shape, self.barcode_matches = None, []
        self.route_nodes, self.route_costs = [], []
        self.obstacle_wait = 0.0
        self.return_wait = 0.0
        self.bypass_gaps = []
        self.capture("准备启动")

    def event(self, kind, message, **extra):
        self.events.append({"t": round(self.t, 3), "task": self.task,
                            "kind": kind, "message": message,
                            "message_en": english_event(kind, message, self.task), **extra})

    def begin(self, task):
        self.task = task
        self.event("task_start", TASKS[task - 1]["label"])
        self.event("qr_task", f"虚拟 QR 任务载荷：{task}（语义传感器，非图像解码）")

    def complete(self):
        self.event("task_complete", f"任务 {self.task} 完成")

    def publish(self, topic, payload):
        message = {"t": round(self.t, 3), "topic": topic, "payload": payload,
                   "transport": "in-process MQTT-shaped simulator bus"}
        self.messages.append(message)
        self.event("message", f"{topic}: {payload}")

    def capture(self, state, cross_track=0):
        rays = []
        for angle in (-math.pi / 3, 0.0, math.pi / 3):
            value, hit = self.world.raycast(*self.robot.truth[:2], self.robot.truth[2] + angle, 2.0)
            rays.append({"angle": angle, "range": round(value, 5), "hit": hit})
        frame = {
            "t": round(self.t, 3), "truth": [round(value, 6) for value in self.robot.truth],
            "odom": [round(value, 6) for value in self.robot.odom], "task": self.task,
            "state": state, "state_en": english_state(state),
            "loaded": self.loaded, "worker_qr": self.worker_qr,
            "obstacle_active": bool(self.active_obstacles), "active_obstacles": list(self.active_obstacles),
            "front_range": rays[1]["range"], "rays": rays, "updates": [], "event": "",
            "command": {"linear": round(self.robot.last_command[0], 5),
                        "angular": round(self.robot.last_command[1], 5)},
            "metrics": {"position_error": round(distance(self.robot.truth, self.robot.odom), 5),
                        "distance": round(self.robot.distance, 4), "mapped_cells": 0,
                        "cross_track": round(cross_track, 6)},
        }
        # A phase can change state without advancing the clock.  Replace the
        # existing sample instead of producing duplicate timestamps, which make
        # continuous playback interpolation ambiguous at the final task boundary.
        if self.frames and self.frames[-1]["t"] == frame["t"]:
            self.frames[-1] = frame
        else:
            self.frames.append(frame)

    def step(self, linear, angular, state, cross_track=0):
        if not self.robot.step(linear, angular, self.dt, self.world):
            raise RuntimeError(f"Collision in {state} at {self.robot.truth}")
        self.steps += 1
        self.t = self.steps * self.dt
        if self.steps % 3 == 0:
            self.capture(state, cross_track)

    def hold(self, duration, state):
        for _ in range(round(duration / self.dt)):
            self.step(0, 0, state)

    def drive(self, points, state, speed=None, odometry=False, callback=None):
        points = [list(point) for point in points]
        if distance(self.robot.truth, points[0]) > 0.04:
            points.insert(0, list(self.robot.truth[:2]))
        self.route.extend(points)
        follower = TrackPIController(points, speed or self.config["robot"]["speed"])
        for _ in range(12000):
            pose = self.robot.odom if odometry else self.robot.truth
            command = follower.command(pose, self.dt)
            if follower.done:
                return
            if callback:
                callback(follower)
            self.step(*command, state, cross_track=follower.cross_track)
        raise RuntimeError(f"Navigation timeout: {state}")

    def edge_path(self, nodes):
        points = []
        lookup = {(edge["from"], edge["to"]): edge["points"] for edge in self.config["warehouse_edges"]}
        for a, b in zip(nodes, nodes[1:]):
            edge = lookup[(a, b)]
            points.extend(edge if not points else edge[1:])
        if not points:
            node = next(value for value in self.config["nodes"] if value["id"] == nodes[0])
            points = [[node["x"], node["y"]]]
        return points

    def task_1(self):
        self.begin(1)
        self.hold(2, "任务 1 · 等待合成交通灯由红变绿")
        self.event("green_light", "合成相机输入：green_fraction 高于阈值")
        self.drive([self.robot.truth[:2], self.config["containers"][0]["scan"]], "任务 1 · PI 沿黑线驶向收货区")
        self.complete()

    def task_2(self):
        self.begin(2)
        requested = self.config["requested_barcode"]
        self.event("barcode_request", f"入口虚拟条码要求：{requested}")
        for container in self.config["containers"]:
            self.drive([self.robot.truth[:2], container["scan"]], "任务 2 · 逐站比较条码")
            self.hold(0.8, f"任务 2 · 读取合成条码 {container['id']}")
            matched = container["id"] == requested
            self.barcode_matches.append({"id": container["id"], "matched": matched, "t": round(self.t, 3)})
            self.event("barcode_match" if matched else "barcode_reject", container["id"])
            if matched:
                approach = [container["x"] - 0.15, container["y"]]
                self.drive([container["scan"], approach], "任务 2 · 叉取已匹配 EC", speed=0.16)
                self.hold(1.0, "任务 2 · 合成叉架提升")
                self.loaded = True
                self.event("fork_lift", f"已装载 {requested}")
                self.drive([approach, container["scan"]], "任务 2 · 返回黑线", speed=0.18)
                break
        if not self.loaded:
            raise RuntimeError("No matching container was available")
        self.complete()

    def task_3(self):
        self.begin(3)
        entry = next(node for node in self.config["nodes"] if node["id"] == "a")
        self.drive([self.robot.truth[:2], [entry["x"], entry["y"]]], "任务 3 · 进入仓储线网")
        inbound, cost_in = graph_shortest_path(self.config["graph"], "a", self.storage)
        outbound, cost_out = graph_shortest_path(self.config["graph"], self.storage, "n")
        self.route_nodes = inbound + outbound[1:]
        self.route_costs = [cost_in, cost_out]
        self.event("plan", " → ".join(inbound), abstract_cost=cost_in)
        self.drive(self.edge_path(inbound), f"任务 3 · Dijkstra 到货位 {self.storage}")
        self.hold(1, "任务 3 · 合成叉架卸货")
        self.loaded = False
        self.event("storage_drop", f"EC 已存入 {self.storage}")
        self.event("plan", " → ".join(outbound), abstract_cost=cost_out)
        self.drive(self.edge_path(outbound), "任务 3 · 沿有向线网离开仓库，出口 n")
        self.complete()

    def task_4(self):
        self.begin(4)
        edge = self.edge_path(["n", "o"])
        # Stop at the blue marker near the end of the curved n->o branch.
        stop = self.config["task4_stop"]
        split = min(range(len(edge)), key=lambda index: distance(edge[index], stop))
        self.drive(edge[:split + 1] + [stop], "任务 4 · 在蓝色标记处到站")
        self.publish("sim/robot/arrival", "arrived")
        self.hold(1.5, "任务 4 · 等待合成工人消息")
        self.packaging_shape = self.config["packaging_shape"]
        self.publish("sim/worker/packaging", self.packaging_shape)
        self.loaded = True
        self.event("worker_container", "工人交付空 EC；合成事件")
        self.drive([stop] + edge[split + 1:], "任务 4 · 离开交互站")
        self.complete()

    def task_5(self):
        self.begin(5)
        path = next(polyline["points"] for polyline in self.config["polylines"] if polyline["id"] == "worker-qr-path")
        self.route.extend(path)
        total, elapsed = polyline_length(path), 0.0
        for _ in range(5000):
            # A synthetic worker advances a manually guided QR at 0.18 m/s.
            self.worker_qr = along_polyline(path, min(total, elapsed * 0.18 + 0.35))
            target = self.worker_qr
            robot_pose = self.robot.truth
            bearing = wrap_angle(math.atan2(target[1] - robot_pose[1], target[0] - robot_pose[0]) - robot_pose[2])
            separation = distance(robot_pose, target)
            linear = min(0.28, max(0.0, 0.8 * (separation - 0.24))) * max(0.0, math.cos(bearing)) ** 2
            angular = max(-1.8, min(1.8, 3.0 * bearing))
            self.step(linear, angular, "任务 5 · 根据合成 QR 方位与距离闭环跟随")
            elapsed += self.dt
            if elapsed * 0.18 + 0.35 >= total and separation < 0.30:
                break
        else:
            raise RuntimeError("Worker QR following did not converge")
        self.worker_qr = None
        self.event("qr_follow_complete", "人工引导 QR 到终点；非无标记人体跟踪")
        self.complete()

    def task_6(self):
        self.begin(6)
        lane = next(polyline["points"] for polyline in self.config["polylines"] if polyline["id"] == "task6-bypass-lane")
        obstacle_config = self.config["task6_obstacle"]
        center = [obstacle_config["x"], obstacle_config["y"]]
        obstacle = Rect(center[0] - obstacle_config["w"] / 2, center[1] - obstacle_config["h"] / 2,
                        obstacle_config["w"], obstacle_config["h"])
        self.world.obstacles = [obstacle]
        self.active_obstacles = ["task6"]
        # Find the closest point on the long diagonal lane segment.
        candidates = []
        for index, (a, b) in enumerate(zip(lane, lane[1:])):
            dx, dy = b[0] - a[0], b[1] - a[1]
            squared = dx * dx + dy * dy
            fraction = max(0, min(1, ((center[0] - a[0]) * dx + (center[1] - a[1]) * dy) / squared)) if squared else 0
            candidate = [a[0] + fraction * dx, a[1] + fraction * dy]
            candidates.append((distance(candidate, center), index, candidate))
        _, segment, center_on_line = min(candidates)
        a, b = lane[segment], lane[segment + 1]
        length = distance(a, b)
        tangent = [(b[0] - a[0]) / length, (b[1] - a[1]) / length]
        normal = [-tangent[1], tangent[0]]

        def local(along, sideways):
            return [center_on_line[0] + tangent[0] * along + normal[0] * sideways,
                    center_on_line[1] + tangent[1] * along + normal[1] * sideways]

        stop = local(-0.43, 0)
        self.drive(lane[:segment + 1] + [stop], "任务 6 · 沿线行驶并检测前方障碍")
        sensed = self.world.raycast(*self.robot.truth[:2], self.robot.truth[2], maximum=2.0)[0]
        if sensed >= 0.50:
            raise RuntimeError("Task 6 requires a real simulated front-range obstacle detection")
        self.event("obstacle_detected", "超声射线检测到障碍，开始 10 秒等待", front_range=sensed)
        began = self.t
        self.hold(obstacle_config["wait_seconds"], "任务 6 · 障碍未离开，等待 10 秒")
        self.obstacle_wait = self.t - began
        self.event("bypass_start", "等待结束，执行几何边界绕障；最大侧距要求 0.30 m")
        detour = [stop, local(-0.31, 0.27), local(0.0, 0.29), local(0.31, 0.27), local(0.48, 0)]

        def record_gap(_):
            relative = [self.robot.truth[0] - center_on_line[0], self.robot.truth[1] - center_on_line[1]]
            along = relative[0] * tangent[0] + relative[1] * tangent[1]
            # The 30 cm side-gap check applies while abreast of the obstacle.
            if abs(along) < 0.13:
                self.bypass_gaps.append(point_rectangle_distance(self.robot.truth, obstacle))

        self.drive(detour, "任务 6 · 近距绕障并重新找回黑线", speed=0.18, callback=record_gap)
        self.world.obstacles, self.active_obstacles = [], []
        self.drive([detour[-1]] + lane[segment + 1:], "任务 6 · 返回黑线并驶向包装区")
        self.complete()

    def task_7(self):
        self.begin(7)
        if self.packaging_shape is None:
            raise RuntimeError("Packaging requires the task 4 worker message")
        matched_station = False
        for station in self.config["packaging"]:
            self.drive([self.robot.truth[:2], station["scan"]], "任务 7 · 沿黑线逐站识别包装形状")
            self.hold(0.7, f"任务 7 · 合成轮廓观察：{station['shape']}")
            matched = station["shape"] == self.packaging_shape
            self.event("shape_match" if matched else "shape_reject", station["shape"])
            if matched:
                matched_station = True
                approach = [station["x"] + 0.15, station["y"]]
                self.drive([station["scan"], approach], "任务 7 · 交付至匹配的包装站", speed=0.16)
                self.hold(1.0, "任务 7 · 合成卸货完成")
                self.loaded = False
                self.drive([approach, station["scan"]], "任务 7 · 返回主线")
                break
        if not matched_station:
            raise RuntimeError("No packing station matches the worker's packaging message")
        self.complete()

    def task_8(self):
        self.begin(8)
        lane = next(polyline["points"] for polyline in self.config["polylines"] if polyline["id"] == "packaging-return-lane")
        # Continue from current packing station rather than resetting to task 7.
        self.drive([self.robot.truth[:2]] + lane[1:], "任务 8 · 沿线返回，直至蓝色终止标记")
        unmarked = next(polyline["points"] for polyline in self.config["polylines"] if polyline["id"] == "charging-unmarked")
        self.event("line_end", "蓝色标记后不再读取线特征，切换编码器里程计导航")
        # The known blue marker anchors a *navigation frame*, without modifying
        # the continuously recorded encoder odometry.  A pose reset here would
        # create a visible jump in the purple trail even though the robot did
        # not teleport.  Apply one fixed SE(2) alignment only to controller input.
        anchor_odom, anchor_truth = list(self.robot.odom), list(self.robot.truth)
        heading_alignment = wrap_angle(anchor_truth[2] - anchor_odom[2])
        alignment_cos, alignment_sin = math.cos(heading_alignment), math.sin(heading_alignment)

        def aligned_odometry():
            dx, dy = self.robot.odom[0] - anchor_odom[0], self.robot.odom[1] - anchor_odom[1]
            return [anchor_truth[0] + alignment_cos * dx - alignment_sin * dy,
                    anchor_truth[1] + alignment_sin * dx + alignment_cos * dy,
                    wrap_angle(self.robot.odom[2] + heading_alignment)]

        self.event("odometry_anchor", "蓝色标记提供导航局部起点；原始编码器轨迹保持连续")
        obstacle_config = self.config["task8_obstacle"]
        center = along_polyline(unmarked, polyline_length(unmarked) * obstacle_config["fraction"])
        obstacle = Rect(center[0] - obstacle_config["w"] / 2, center[1] - obstacle_config["h"] / 2,
                        obstacle_config["w"], obstacle_config["h"])
        self.world.obstacles, self.active_obstacles = [obstacle], ["task8"]
        follower = TrackPIController([list(anchor_truth[:2]), unmarked[-1]], speed=0.23)
        self.route.extend(unmarked)
        wait_started, cleared = None, False
        for _ in range(5000):
            front, _ = self.world.raycast(*self.robot.truth[:2], self.robot.truth[2], maximum=2.0)
            if not cleared and (front < 0.4 or wait_started is not None):
                if wait_started is None:
                    wait_started = self.t
                    self.event("return_obstacle", "无标线路段遇障：停车，等待障碍自行离开")
                if self.t - wait_started >= obstacle_config["clear_after_seconds"]:
                    self.return_wait = self.t - wait_started
                    self.world.obstacles, self.active_obstacles, cleared = [], [], True
                    self.event("return_obstacle_cleared", "合成障碍已离开，继续编码器导航")
                else:
                    self.step(0, 0, "任务 8 · 等待障碍离开，不绕行")
                    continue
            command = follower.command(aligned_odometry(), self.dt)
            if follower.done:
                break
            self.step(*command, "任务 8 · 无标线编码器导航至充电站", cross_track=follower.cross_track)
        else:
            raise RuntimeError("Odometry return did not converge")
        self.event("charging", "已到充电站；仿真停车，不涉及真实电气充电")
        self.complete()

    def run(self):
        for task in range(1, 9):
            getattr(self, f"task_{task}")()
        self.robot.last_command = [0, 0]
        self.capture("八项原始 Challenge 任务完成")
        config = self.config
        obstacle6 = config["task6_obstacle"]
        unmarked = next(polyline["points"] for polyline in config["polylines"] if polyline["id"] == "charging-unmarked")
        obstacle8_center = along_polyline(unmarked, polyline_length(unmarked) * config["task8_obstacle"]["fraction"])
        obstacle8 = config["task8_obstacle"]
        obstacles = [{"id": "task6", "x": obstacle6["x"] - obstacle6["w"] / 2,
                      "y": obstacle6["y"] - obstacle6["h"] / 2, "w": obstacle6["w"], "h": obstacle6["h"]},
                     {"id": "task8", "x": obstacle8_center[0] - obstacle8["w"] / 2,
                      "y": obstacle8_center[1] - obstacle8["h"] / 2, "w": obstacle8["w"], "h": obstacle8["h"]}]
        stations = [{"x": marker["x"], "y": marker["y"], "label": str(marker["task"]), "kind": "task"}
                    for marker in config["markers"]]
        stations.append({"x": config["charger"]["x"], "y": config["charger"]["y"], "label": "CHARGE", "kind": "goal"})
        return {
            "version": 1, "id": "original-challenge", "title": "原始课程 Challenge · 可运行仿真重构",
            "title_en": "Original Course Challenge - Runnable Simulation Reconstruction",
            "scenario": "original_challenge", "seed": self.seed, "noise": "low",
            "world": {"width": config["width"], "height": config["height"], "obstacles": obstacles,
                      "stations": stations, "route": self.route},
            "grid": {"width": 36, "height": 36, "resolution": 0.2},
            "track": {key: config[key] for key in ("polylines", "nodes", "markers", "blue_markers", "charger",
                                                       "containers", "packaging", "scale_note")},
            "tasks": TASKS, "frames": self.frames, "events": self.events, "synthetic_messages": self.messages,
            "metrics": {"completed": True, "duration": round(self.t, 2), "distance": round(self.robot.distance, 4),
                        "collisions": self.robot.collisions, "final_position_error": round(distance(self.robot.truth, self.robot.odom), 5),
                        "charging_error": round(distance(self.robot.truth, [config["charger"]["x"], config["charger"]["y"]]), 5),
                        "position_rmse": round(math.sqrt(sum(frame["metrics"]["position_error"] ** 2 for frame in self.frames) / len(self.frames)), 5),
                        "known_fraction": 0, "task6_wait_seconds": round(self.obstacle_wait, 3),
                        "task8_wait_seconds": round(self.return_wait, 3), "task6_max_side_gap": round(max(self.bypass_gaps), 5),
                        "task6_min_side_gap": round(min(self.bypass_gaps), 5)},
            "warehouse_plan": {"storage": self.storage, "nodes": self.route_nodes, "abstract_costs": self.route_costs},
            "barcode_checks": self.barcode_matches,
            "notes_en": [config["scale_note"],
                         "The 2025-09 hardware course completion is separate from this newly written 2026 simulation.",
                         "Original eight-task semantics and line topology; camera/QR/shape/worker observations are explicit synthetic inputs.",
                         "Black-line segments use virtual line features and PI/heading control. The unmarked return uses encoder odometry.",
                         "The blue marker anchors a fixed navigation frame; exported raw encoder odometry is continuous and never reset.",
                         "Bypass uses known obstacle geometry; messaging is a local bus rather than network MQTT. This is not a full SLAM implementation."],
            "notes": [config["scale_note"],
                      "2025.09 原课程真实机器人实践已由本人确认完成；此代码为 2026 年重新编写的仿真版本。",
                      "保留原八项任务和黑线拓扑；相机/QR/形状与工人消息使用显式合成输入，不是已实现的图像解码。",
                      "黑线段：几何虚拟线特征 + PI/航向控制；最后无标线段只用编码器里程计，非完整 SLAM。",
                      "绕障侧距按机器人中心至障碍边界计算；机器人半径 0.10 m，实际外壳间隙更小。",
                      "绕障重构使用已知障碍几何，不声称未知环境自主 BUG2；消息为本地总线，不是网络 MQTT。"],
        }


def simulate_original_challenge(storage="i", seed=42, config_path=None):
    return OriginalChallenge(load_map(config_path), storage, seed).run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", choices=("g", "i", "k", "l"), default="i")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "data" / "original-challenge.json")
    args = parser.parse_args()
    run = simulate_original_challenge(args.storage, args.seed, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(run, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "frames": len(run["frames"]), "metrics": run["metrics"],
                      "warehouse_plan": run["warehouse_plan"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
