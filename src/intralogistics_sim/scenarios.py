"""Deterministic synthetic scenes exported as actual simulation playback data."""

import math
import random

from .geometry import World, Rect, distance, warehouse_world
from .robot import DifferentialDrive, LOW_NOISE, HIGH_NOISE
from .mapping import OccupancyGrid
from .planning import WaypointFollower, GeometricBug2, dijkstra


SENSOR_ANGLES = [math.radians(value) for value in (-75, -50, -25, 0, 25, 50, 75)]


def scan(robot, world, rng):
    rays = []
    for angle in SENSOR_ANGLES:
        value, hit = world.raycast(*robot.truth[:2], robot.truth[2] + angle, maximum=3.1)
        # Synthetic ultrasonic measurement noise is identical across odometry comparisons.
        value = max(0.02, min(3.1, value + rng.gauss(0, 0.008)))
        rays.append({"angle": round(angle, 6), "range": round(value, 5), "hit": hit})
    return rays


class Recorder:
    def __init__(self, run_id, title, scenario, noise, world, start, stations, route, tasks, seed=42,
                 encoder_model=None):
        self.robot = DifferentialDrive(start, encoder_model or (HIGH_NOISE if noise == "high" else LOW_NOISE), seed)
        self.grid = OccupancyGrid(world)
        self.world, self.sensor_random = world, random.Random(seed + 1000)
        self.frames, self.errors, self.events = [], [], []
        self.run = {
            "version": 1, "id": run_id, "title": title, "scenario": scenario,
            "noise": noise, "seed": seed, "world": world.to_dict(stations, route),
            "grid": self.grid.metadata(), "tasks": tasks,
            "notes": [
                "2026 年新写的课程主题仿真重构；不是遗失的 2025 年课程源码。",
                "路线使用已知布局；运动控制读取仿真参考位姿，地图只读取编码器里程计。",
                "里程计驱动的占据栅格建图：没有观测位姿校正、回环或全局优化。",
                "播放由 Python 计算的记录；传感器、场景和任务观察均为合成数据。",
            ],
        }

    def event(self, t, task, kind, message):
        self.events.append({"t": round(t, 3), "task": task, "kind": kind, "message": message})

    def capture(self, t, task, state, event=""):
        rays = scan(self.robot, self.world, self.sensor_random)
        updates = self.grid.update(self.robot.odom, rays)
        error = distance(self.robot.truth, self.robot.odom)
        self.errors.append(error)
        self.frames.append({
            "t": round(t, 3), "truth": [round(value, 6) for value in self.robot.truth],
            "odom": [round(value, 6) for value in self.robot.odom],
            "task": task, "state": state, "rays": rays, "updates": updates, "event": event,
            "metrics": {"position_error": round(error, 5), "mapped_cells": len(self.grid.observed),
                        "distance": round(self.robot.distance, 4)},
            "command": {"linear": round(self.robot.last_command[0], 4),
                        "angular": round(self.robot.last_command[1], 4)},
        })

    def finish(self, completed, duration):
        self.run.update({
            "frames": self.frames, "events": self.events,
            "metrics": {
                "completed": completed, "duration": round(duration, 2),
                "distance": round(self.robot.distance, 4),
                "position_rmse": round(math.sqrt(sum(value * value for value in self.errors) / len(self.errors)), 5),
                "final_position_error": round(self.errors[-1], 5),
                "known_fraction": round(self.grid.known_fraction, 5),
                "collisions": self.robot.collisions,
            },
        })
        return self.run


def make_route(world, stops):
    path = [list(stops[0])]
    for start, goal in zip(stops, stops[1:]):
        path.extend(dijkstra(world, start, goal)[1:])
    return path


def simulate_mapping(noise="low", seed=42, encoder_model=None):
    world = warehouse_world()
    stops = [[1.0, 1.0], [8.9, 1.0], [8.9, 5.9], [4.8, 5.9],
             [4.8, 1.0], [1.0, 5.8], [1.0, 1.0]]
    route = make_route(world, stops)
    run_id = "mapping-noisy" if noise == "high" else "mapping-clean"
    title = "迷宫建图 · 里程计偏差" if noise == "high" else "迷宫建图 · 低噪声"
    rec = Recorder(run_id, title, "mapping", noise, world, [*stops[0], 0],
                   [{"x": 1, "y": 1, "label": "START / END", "kind": "start"}],
                   route, [{"id": 1, "label": "按已知路线采样并建图"}], seed, encoder_model)
    follower = WaypointFollower(route)
    rec.event(0, 1, "state", "开始：参考位姿导航，编码器里程计独立建图")
    rec.capture(0, 1, "沿已知布局采样")
    steps = 0
    for steps in range(1, 4001):
        command = follower.command(rec.robot.truth)
        rec.robot.step(*command, 0.1, world)
        if steps % 2 == 0 or follower.done:
            rec.capture(steps * 0.1, 1, "采样完成" if follower.done else "沿已知布局采样")
        if follower.done:
            break
    rec.event(steps * 0.1, 1, "complete", "路线完成；累计里程计误差保留，未做回环校正")
    return rec.finish(follower.done, steps * 0.1)


class SyntheticMessageBus:
    """In-process event bus, intentionally not represented as a real MQTT broker."""
    def __init__(self):
        self.messages = []

    def publish(self, topic, payload, t):
        self.messages.append({"topic": topic, "payload": payload, "t": round(t, 3)})


TASKS = [
    {"id": 1, "label": "绿灯启动 · 合成信号"},
    {"id": 2, "label": "驶向识别站 · 已知路线"},
    {"id": 3, "label": "颜色 / 形状识别 · 合成观察"},
    {"id": 4, "label": "标记站靠近 · 合成位姿目标"},
    {"id": 5, "label": "装载确认 · 合成事件"},
    {"id": 6, "label": "Dijkstra 规划与配送"},
    {"id": 7, "label": "条码交接 · 本地消息总线"},
    {"id": 8, "label": "返回起点并停止"},
]


def simulate_challenge(seed=42):
    world = warehouse_world()
    start, pickup, dock, delivery = [1.0, 1.0], [2.05, 1.0], [2.05, 1.35], [8.8, 5.95]
    nav_goals = {2: pickup, 4: dock, 6: delivery, 8: start}
    route = make_route(world, [start, pickup, dock, delivery, start])
    stations = [
        {"x": start[0], "y": start[1], "label": "START", "kind": "start"},
        {"x": pickup[0], "y": pickup[1], "label": "SCAN", "kind": "sensor"},
        {"x": dock[0], "y": dock[1], "label": "PICKUP", "kind": "pickup"},
        {"x": delivery[0], "y": delivery[1], "label": "DELIVERY", "kind": "delivery"},
    ]
    rec = Recorder("challenge", "八步物流任务 · 合成场景", "challenge", "low", world,
                   [*start, 0.0], stations, route, TASKS, seed)
    rec.run["notes"].extend([
        "八步任务是新写的课程主题重构，不声称逐项复现原 Challenge 验收。",
        "颜色、标记、条码和装载由合成观察触发；未实现图像 QR 解码或网络 MQTT。",
        "P / PI 实际闭环循线对比在单独实验中；本场景移动使用参考位姿路径控制。",
    ])
    bus = SyntheticMessageBus()
    task, entered, follower = 1, 0.0, None
    completed, t = False, 0.0
    rec.event(0, 1, "state", TASKS[0]["label"])
    rec.capture(0, 1, TASKS[0]["label"])
    for step in range(1, 4001):
        t = step * 0.1
        done, observation = False, ""
        if task in nav_goals:
            if follower is None:
                follower = WaypointFollower(dijkstra(world, rec.robot.truth, nav_goals[task]),
                                            speed=0.3 if task == 4 else 0.48)
            command = follower.command(rec.robot.truth)
            rec.robot.step(*command, 0.1, world)
            done = follower.done
        else:
            # Hold the real plant at rest until an explicitly synthetic observation.
            rec.robot.step(0.0, 0.0, 0.1, world)
            delay = {1: 2.0, 3: 1.4, 5: 1.3, 7: 1.3}[task]
            if t - entered >= delay - 1e-8:
                done = True
                observation = {
                    1: "合成信号：green",
                    3: "合成观察：blue rectangular parcel",
                    5: "合成装载传感器：loaded = true",
                    7: "合成条码：SYNTHETIC-DELIVERY-A；本地总线确认交接",
                }[task]
                bus.publish(f"synthetic/task/{task}", observation, t)
                rec.event(t, task, "observation", observation)
        if done:
            rec.event(t, task, "complete", f"步骤 {task} 完成")
            if task == 8:
                completed = True
            else:
                task += 1
                entered, follower = t, None
                rec.event(t, task, "state", TASKS[task - 1]["label"])
        if step % 2 == 0 or completed:
            rec.capture(t, task, "全部任务完成" if completed else TASKS[task - 1]["label"], observation)
        if completed:
            break
    rec.run["synthetic_messages"] = bus.messages
    return rec.finish(completed, t)


def simulate_bug2(seed=42, encoder_model=None):
    obstacle = Rect(4.0, 2.2, 1.5, 2.5)
    world = World(10.0, 7.0, [obstacle])
    start, goal = [1.0, 3.5], [9.0, 3.5]
    rec = Recorder("bug2", "BUG2 原理 · M 线与边界跟随", "bug2", "low", world,
                   [*start, 0.0],
                   [{"x": 1, "y": 3.5, "label": "START", "kind": "start"},
                    {"x": 9, "y": 3.5, "label": "GOAL", "kind": "goal"}],
                   [start, goal], [{"id": 1, "label": "沿 M 线 → 边界跟随 → 更近的离开点"}], seed,
                   encoder_model)
    rec.run["notes"].append(
        "BUG2 教学重构：前向测距触发绕障；用矩形已知几何及参考位姿跟随膨胀边界，非未知环境自主 BUG2。"
    )
    controller = GeometricBug2(start, goal, obstacle)
    last_state = controller.state
    rec.event(0, 1, "state", controller.state)
    rec.capture(0, 1, controller.state)
    step = 0
    for step in range(1, 2201):
        front_range, _ = world.raycast(*rec.robot.truth[:2], rec.robot.truth[2], maximum=3.1)
        command = controller.command(rec.robot.truth, front_range)
        rec.robot.step(*command, 0.1, world)
        if controller.state != last_state:
            rec.event(step * 0.1, 1, "state", controller.state)
            last_state = controller.state
        if step % 2 == 0 or controller.done:
            rec.capture(step * 0.1, 1, controller.state)
        if controller.done:
            break
    rec.run["bug2"] = {"hit_distance_to_goal": controller.hit_distance,
                       "leave_distance_to_goal": controller.leave_distance,
                       "transitions": controller.transitions}
    return rec.finish(controller.done, step * 0.1)


def simulate_line_following(seed=42):
    """Kinematic line-sensor feedback under a constant yaw-rate disturbance.

    Both controllers use P heading damping.  PI additionally integrates measured
    lateral error.  Neither controller uses the unknown disturbance in control.
    """
    runs = []
    dt, speed, bias, duration = 0.04, 0.3, 0.10, 40.0
    for name, integral_gain in (("p", 0.0), ("pi", 0.55)):
        x, y, theta, integral = 0.0, 0.25, 0.0, 0.0
        rng = random.Random(seed)
        samples = []
        for step in range(round(duration / dt) + 1):
            measured = y + rng.gauss(0.0, 0.0015)
            integral = max(-0.8, min(0.8, integral + measured * dt))
            control = max(-1.5, min(1.5, -2.0 * measured - 2.3 * theta - integral_gain * integral))
            samples.append({"t": round(step * dt, 4), "error": round(y, 6),
                            "heading": round(theta, 6), "control": round(control, 6),
                            "x": round(x, 6), "y": round(y, 6), "reference_y": 0.0})
            # The same differential-drive arc integrator as the warehouse plant.
            from .robot import integrate_pose
            x, y, theta = integrate_pose([x, y, theta],
                                        (speed - (control + bias) * 0.34 / 2) * dt,
                                        (speed + (control + bias) * 0.34 / 2) * dt, 0.34)
        tail = samples[len(samples) * 3 // 4:]
        runs.append({"id": name, "label": name.upper(), "samples": samples,
                     "metrics": {"rms_error": round(math.sqrt(sum(s["error"] ** 2 for s in samples) / len(samples)), 6),
                                 "final_error": samples[-1]["error"],
                                 "steady_rms_error": round(math.sqrt(sum(s["error"] ** 2 for s in tail) / len(tail)), 6)}})
    return {"version": 1, "seed": seed, "runs": runs,
            "reference": {"length": speed * duration, "initial_offset": 0.25,
                          "speed": speed, "steering_bias": bias, "duration": duration},
            "notes": ["合成直线传感器与恒定偏航扰动；比较 P 和 PI 横向闭环控制，不是 PID / PD。",
                      "两者都使用航向阻尼；PI 额外积累横向误差。噪声、初态、速度与扰动保持一致。"]}
