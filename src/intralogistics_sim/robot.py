"""Differential-drive plant, incremental wheel encoders and pose integration."""

from dataclasses import dataclass
import math
import random

from .geometry import wrap_angle


def integrate_pose(pose, left_distance, right_distance, wheelbase):
    """Exact constant-wheel-velocity arc, with stable straight-line limit."""
    x, y, theta = pose
    ds = (right_distance + left_distance) / 2
    da = (right_distance - left_distance) / wheelbase
    if abs(da) < 1e-10:
        step = ds
    else:
        step = ds * math.sin(da / 2) / (da / 2)
    return [x + step * math.cos(theta + da / 2),
            y + step * math.sin(theta + da / 2), wrap_angle(theta + da)]


@dataclass(frozen=True)
class EncoderModel:
    left_scale: float = 1.0
    right_scale: float = 1.0
    wheelbase_scale: float = 1.0
    distance_noise: float = 0.0
    resolution: float = 0.0005


LOW_NOISE = EncoderModel(1.0002, 0.9998, 1.0001, 0.000025)
HIGH_NOISE = EncoderModel(1.003, 0.9985, 1.003, 0.0001)


class DifferentialDrive:
    def __init__(self, pose, noise=LOW_NOISE, seed=42, wheelbase=0.34):
        self.truth, self.odom = list(pose), list(pose)
        self.noise, self.wheelbase = noise, wheelbase
        self.random = random.Random(seed)
        self.radius = 0.18
        self.distance = 0.0
        self.collisions = 0
        self._totals = [0.0, 0.0]
        self._encoder_totals = [0.0, 0.0]
        self.last_command = [0.0, 0.0]

    def step(self, linear, angular, dt, world):
        left = (linear - angular * self.wheelbase / 2) * dt
        right = (linear + angular * self.wheelbase / 2) * dt
        candidate = integrate_pose(self.truth, left, right, self.wheelbase)
        if world.collides(candidate[0], candidate[1], self.radius):
            self.collisions += 1
            self.last_command = [0.0, 0.0]
            return False
        self.truth = candidate
        self.distance += abs((left + right) / 2)
        measured = []
        for index, delta in enumerate((left, right)):
            self._totals[index] += delta
            total = round(self._totals[index] / self.noise.resolution) * self.noise.resolution
            increment = total - self._encoder_totals[index]
            self._encoder_totals[index] = total
            scale = self.noise.left_scale if index == 0 else self.noise.right_scale
            # Independent incremental encoder noise; never add noise to the truth.
            measured.append(increment * scale + self.random.gauss(0, self.noise.distance_noise))
        self.odom = integrate_pose(
            self.odom, measured[0], measured[1],
            self.wheelbase * self.noise.wheelbase_scale,
        )
        self.last_command = [linear, angular]
        return True
