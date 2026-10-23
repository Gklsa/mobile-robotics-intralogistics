"""Odometry-fed inverse range sensor mapping; no loop closure or pose correction."""

import math


def bresenham(x0, y0, x1, y1):
    cells = []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx + dy
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return cells
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy


class OccupancyGrid:
    def __init__(self, world, resolution=0.2):
        self.resolution = resolution
        self.width = math.ceil(world.width / resolution)
        self.height = math.ceil(world.height / resolution)
        self.log_odds = [0] * (self.width * self.height)
        self.observed = set()
        self.values = [-1] * len(self.log_odds)

    def metadata(self):
        return {"width": self.width, "height": self.height, "resolution": self.resolution}

    def update(self, pose, rays):
        changes = {}
        start_x, start_y = math.floor(pose[0] / self.resolution), math.floor(pose[1] / self.resolution)
        for ray in rays:
            angle = pose[2] + ray["angle"]
            end_x = math.floor((pose[0] + math.cos(angle) * ray["range"]) / self.resolution)
            end_y = math.floor((pose[1] + math.sin(angle) * ray["range"]) / self.resolution)
            cells = bresenham(start_x, start_y, end_x, end_y)
            for step, (x, y) in enumerate(cells):
                if not (0 <= x < self.width and 0 <= y < self.height):
                    continue
                index = y * self.width + x
                occupied = ray["hit"] and step == len(cells) - 1
                self.log_odds[index] = max(-10, min(10, self.log_odds[index] + (4 if occupied else -1)))
                self.observed.add(index)
                value = 100 if self.log_odds[index] > 0 else 0
                if self.values[index] != value:
                    self.values[index] = value
                    changes[index] = value
        return [[index, value] for index, value in sorted(changes.items())]

    @property
    def known_fraction(self):
        return len(self.observed) / len(self.values)
