"""Known-layout Dijkstra navigation and a geometric, course-inspired BUG2 demo."""

import heapq
import math

from .geometry import distance, wrap_angle


def dijkstra(world, start, goal, resolution=0.2, clearance=0.28):
    """Plan on an inflated known-layout grid with collision-checked connectors.

    This is deliberately not navigation on the progressively built sensor map.
    Diagonal corner cutting is rejected by checking every candidate edge.
    """
    width, height = math.ceil(world.width / resolution), math.ceil(world.height / resolution)

    def point(cell):
        return ((cell[0] + 0.5) * resolution, (cell[1] + 0.5) * resolution)

    def nearest(pose):
        raw = (int(pose[0] / resolution), int(pose[1] / resolution))
        candidates = [(x, y) for x in range(max(0, raw[0] - 2), min(width, raw[0] + 3))
                      for y in range(max(0, raw[1] - 2), min(height, raw[1] + 3))]
        valid = [cell for cell in candidates if world.segment_clear(pose, point(cell), clearance)]
        if not valid:
            raise ValueError("No collision-free endpoint connector")
        return min(valid, key=lambda cell: distance(point(cell), pose))

    first, last = nearest(start), nearest(goal)
    costs, parents, queue = {first: 0.0}, {}, [(0.0, first)]
    while queue:
        cost, cell = heapq.heappop(queue)
        if cost != costs[cell]:
            continue
        if cell == last:
            break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            neighbor = (cell[0] + dx, cell[1] + dy)
            if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                continue
            if not world.segment_clear(point(cell), point(neighbor), clearance):
                continue
            candidate = cost + resolution * math.hypot(dx, dy)
            if candidate + 1e-12 < costs.get(neighbor, math.inf):
                costs[neighbor] = candidate
                parents[neighbor] = cell
                heapq.heappush(queue, (candidate, neighbor))
    if last not in costs:
        raise ValueError("No known-layout route exists")
    cells, current = [last], last
    while current != first:
        current = parents[current]
        cells.append(current)
    raw = [list(start[:2])] + [list(point(cell)) for cell in reversed(cells)] + [list(goal[:2])]
    # Greedy visibility simplification preserves clearance, unlike merely dropping nodes.
    simplified, index = [raw[0]], 0
    while index < len(raw) - 1:
        next_index = len(raw) - 1
        while not world.segment_clear(raw[index], raw[next_index], clearance):
            next_index -= 1
        simplified.append(raw[next_index])
        index = next_index
    return simplified


class WaypointFollower:
    """Reference-pose steering for repeatable mapping comparison.

    The simulator's reference pose is exposed to this controller.  Encoder
    odometry is independently used for mapping and its drift is never reset.
    This separation is explicit in export notes and is not an SLAM solution.
    """
    def __init__(self, path, speed=0.48, tolerance=0.07):
        self.path = path
        self.index = 1 if len(path) > 1 else 0
        self.speed, self.tolerance = speed, tolerance

    def command(self, pose):
        while self.index < len(self.path) and distance(pose, self.path[self.index]) < self.tolerance:
            self.index += 1
        if self.done:
            return 0.0, 0.0
        target = self.path[self.index]
        error = wrap_angle(math.atan2(target[1] - pose[1], target[0] - pose[0]) - pose[2])
        angular = max(-1.6, min(1.6, 2.8 * error))
        linear = min(self.speed, 1.2 * distance(pose, target)) * max(0.0, math.cos(error)) ** 3
        return linear, angular

    @property
    def done(self):
        return self.index >= len(self.path)


class GeometricBug2:
    """M-line / boundary / leave-point state machine in one synthetic rectangle.

    Front range triggers obstacle contact; the boundary follower uses the
    rectangle's inflated geometric boundary and a reference pose.  It is a
    transparent BUG2 teaching reconstruction, not an unknown-world deployment.
    """
    def __init__(self, start, goal, obstacle):
        self.start, self.goal, self.obstacle = start, goal, obstacle
        self.state = "沿 M 线前进"
        self.hit_distance = None
        self.leave_distance = None
        self.follower = WaypointFollower([list(start), list(goal)], speed=0.4)
        self.transitions = []

    def command(self, pose, front_range):
        if self.state == "沿 M 线前进" and front_range < 0.67:
            self.hit_distance = distance(pose, self.goal)
            margin = 0.45
            left, right = self.obstacle.x - margin, self.obstacle.x + self.obstacle.w + margin
            bottom = self.obstacle.y - margin
            self.follower = WaypointFollower([
                list(pose[:2]), [left, bottom], [right, bottom], [right, self.goal[1]],
            ], speed=0.34, tolerance=0.045)
            self.state = "绕障边界跟随"
            self.transitions.append(self.state)
        if self.state == "绕障边界跟随":
            # Leave only after following the boundary to a closer point on M-line.
            on_m_line = abs(pose[1] - self.goal[1]) < 0.055
            closer = distance(pose, self.goal) < self.hit_distance - 0.2
            past_obstacle = pose[0] > self.obstacle.x + self.obstacle.w + 0.3
            if on_m_line and closer and past_obstacle:
                self.leave_distance = distance(pose, self.goal)
                self.state = "返回 M 线"
                self.transitions.append(self.state)
                self.follower = WaypointFollower([list(pose[:2]), list(self.goal)], speed=0.4)
        command = self.follower.command(pose)
        if self.state == "返回 M 线" and self.follower.done:
            self.state = "到达终点"
            self.transitions.append(self.state)
        return command

    @property
    def done(self):
        return self.state == "到达终点"
