"""Metric 2-D geometry, ray casting and a circular-robot collision model."""

from dataclasses import dataclass
import math


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    def to_dict(self):
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def intersects_circle(self, x, y, radius):
        near_x = max(self.x, min(x, self.x + self.w))
        near_y = max(self.y, min(y, self.y + self.h))
        return (x - near_x) ** 2 + (y - near_y) ** 2 <= radius ** 2

    def ray_distance(self, x, y, dx, dy):
        """Slab intersection: nearest nonnegative distance, or infinity."""
        near, far = -math.inf, math.inf
        for origin, direction, low, high in (
            (x, dx, self.x, self.x + self.w),
            (y, dy, self.y, self.y + self.h),
        ):
            if abs(direction) < 1e-12:
                if not low <= origin <= high:
                    return math.inf
                continue
            a, b = (low - origin) / direction, (high - origin) / direction
            near, far = max(near, min(a, b)), min(far, max(a, b))
            if near > far:
                return math.inf
        if far < 0:
            return math.inf
        return max(0.0, near)


@dataclass
class World:
    width: float
    height: float
    obstacles: list

    def collides(self, x, y, radius=0.18):
        return (
            x < radius or y < radius
            or x > self.width - radius or y > self.height - radius
            or any(rect.intersects_circle(x, y, radius) for rect in self.obstacles)
        )

    def segment_clear(self, start, end, clearance=0.25):
        steps = max(1, math.ceil(distance(start, end) / 0.045))
        return all(
            not self.collides(
                start[0] + (end[0] - start[0]) * i / steps,
                start[1] + (end[1] - start[1]) * i / steps,
                clearance,
            )
            for i in range(steps + 1)
        )

    def raycast(self, x, y, angle, maximum=3.0):
        """Return distance/hit from walls and rectangles, not visual placeholders."""
        dx, dy = math.cos(angle), math.sin(angle)
        hits = [rect.ray_distance(x, y, dx, dy) for rect in self.obstacles]
        if dx > 1e-12:
            hits.append((self.width - x) / dx)
        elif dx < -1e-12:
            hits.append(-x / dx)
        if dy > 1e-12:
            hits.append((self.height - y) / dy)
        elif dy < -1e-12:
            hits.append(-y / dy)
        nearest = min((value for value in hits if value >= 0), default=math.inf)
        return min(maximum, nearest), nearest <= maximum

    def to_dict(self, stations=None, route=None):
        return {
            "width": self.width, "height": self.height,
            "obstacles": [rect.to_dict() for rect in self.obstacles],
            "stations": stations or [], "route": route or [],
        }


def warehouse_world():
    return World(10.0, 7.0, [
        Rect(3.0, 1.7, 0.65, 3.0),
        Rect(6.1, 2.3, 0.65, 3.2),
        Rect(7.7, 1.8, 0.8, 0.65),
    ])
