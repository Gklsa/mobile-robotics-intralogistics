#!/usr/bin/env python3
"""Native, perspective 3D studio for the original course Challenge recording.

The mesh is newly authored. Panda3D renders the supplied simulation frames; it
does not replace their dynamics. There is no network or robot connection.
Run ``python native_3d.py`` or export with ``--render-frame image.png``.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
for local_runtime in [ROOT / '.runtime', ROOT.parent / 'panda_runtime']:
    if local_runtime.is_dir():
        sys.path.insert(0, str(local_runtime))

try:
    from panda3d.core import (
        AmbientLight, CardMaker, DirectionalLight, Filename, Fog, Geom,
        GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat,
        GeomVertexWriter, LineSegs, Material, NodePath, PNMImage, Point3,
        PerspectiveLens, TextNode, Texture, TransparencyAttrib, Vec3, Vec4, loadPrcFileData,
    )
    from direct.showbase.ShowBase import ShowBase
    from direct.gui.DirectGui import DirectButton, DirectCheckButton, DirectFrame, DirectLabel, DirectSlider
    from direct.gui import DirectGuiGlobals
except ImportError as exc:
    raise ImportError('3D Studio requires Panda3D. Run: python -m pip install -r requirements-3d.txt') from exc


TASK_LABELS = [
    'Wait for green light', 'Barcode match & pickup', 'Dijkstra storage route',
    'Packaging message', 'Follow moving QR', 'Wait 10 s & bypass',
    'Shape-match & unload', 'Encoder return to charge',
]
TASK_SHORT = ['SIGNAL', 'PICKUP', 'STORAGE', 'MESSAGE', 'QR FOLLOW', 'BYPASS', 'UNLOAD', 'RETURN']
INK = (0.075, 0.14, 0.19, 1)
MUTED = (0.36, 0.45, 0.50, 1)
TEAL = (0.02, 0.49, 0.51, 1)
BLUE = (0.18, 0.38, 0.74, 1)
PAPER = (0.94, 0.96, 0.955, 1)
WHITE = (0.99, 0.985, 0.96, 1)
BLACK = (0.045, 0.055, 0.065, 1)
RED = (0.75, 0.14, 0.14, 1)


def load_recording(path=None):
    if isinstance(path, dict):
        run = path
    else:
        path = Path(path) if path else ROOT / 'docs/data/original-challenge.json'
        if not path.exists():
            raise FileNotFoundError('Generate the recording: python tools/generate_original_demo.py')
        run = json.loads(path.read_text(encoding='utf-8'))
    if not run.get('frames') or not run.get('track', {}).get('polylines'):
        raise ValueError('A recording with frames and the original Challenge track is required.')
    return run


def interpolated_frame(run, times, t):
    """Interpolate only continuous recorded values; keep task events discrete."""
    frames = run['frames']
    i = max(0, min(len(frames) - 1, bisect.bisect_right(times, t) - 1))
    a, b = frames[i], frames[min(i + 1, len(frames) - 1)]
    span = b['t'] - a['t']
    u = max(0, min(1, (t - a['t']) / span)) if span else 0
    result = dict(a)
    for key in ('truth', 'odom'):
        p, q = a[key], b[key]
        # Heading crosses +/-pi; interpolate the shortest angular arc.
        da = (q[2] - p[2] + math.pi) % (2 * math.pi) - math.pi
        result[key] = [p[0] + u * (q[0] - p[0]), p[1] + u * (q[1] - p[1]), p[2] + u * da]
    result['t'] = t
    result['metrics'] = dict(a.get('metrics', {}))
    for key in ('distance', 'position_error'):
        if key in a.get('metrics', {}) and key in b.get('metrics', {}):
            result['metrics'][key] = a['metrics'][key] + u * (b['metrics'][key] - a['metrics'][key])
    return result, i


def mesh_node(name, points, normals, triangles):
    data = GeomVertexData(name, GeomVertexFormat.getV3n3(), Geom.UHStatic)
    vertices, vertex_normals = GeomVertexWriter(data, 'vertex'), GeomVertexWriter(data, 'normal')
    for point, normal in zip(points, normals):
        vertices.addData3(*point)
        vertex_normals.addData3(*normal)
    primitive = GeomTriangles(Geom.UHStatic)
    for triangle in triangles:
        primitive.addVertices(*triangle)
    primitive.closePrimitive()
    geom = Geom(data)
    geom.addPrimitive(primitive)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)


def box_mesh():
    faces = [
        ((0, 0, 1), [(-.5, -.5, .5), (.5, -.5, .5), (.5, .5, .5), (-.5, .5, .5)]),
        ((0, 0, -1), [(-.5, .5, -.5), (.5, .5, -.5), (.5, -.5, -.5), (-.5, -.5, -.5)]),
        ((0, -1, 0), [(-.5, -.5, -.5), (.5, -.5, -.5), (.5, -.5, .5), (-.5, -.5, .5)]),
        ((0, 1, 0), [(.5, .5, -.5), (-.5, .5, -.5), (-.5, .5, .5), (.5, .5, .5)]),
        ((1, 0, 0), [(.5, -.5, -.5), (.5, .5, -.5), (.5, .5, .5), (.5, -.5, .5)]),
        ((-1, 0, 0), [(-.5, .5, -.5), (-.5, -.5, -.5), (-.5, -.5, .5), (-.5, .5, .5)]),
    ]
    points, normals, triangles = [], [], []
    for normal, corners in faces:
        start = len(points)
        points.extend(corners)
        normals.extend([normal] * 4)
        triangles.extend([(start, start + 1, start + 2), (start, start + 2, start + 3)])
    return mesh_node('unit-box', points, normals, triangles)


def cylinder_mesh(sides=32):
    points, normals, triangles = [], [], []
    for i in range(sides + 1):
        angle = i / sides * math.tau
        x, y = math.cos(angle), math.sin(angle)
        for z in (-.5, .5):
            points.append((x, y, z))
            normals.append((x, y, 0))
    for i in range(sides):
        k = i * 2
        triangles.extend([(k, k + 2, k + 1), (k + 1, k + 2, k + 3)])
    for z, sign in ((-.5, -1), (.5, 1)):
        center = len(points)
        points.append((0, 0, z))
        normals.append((0, 0, sign))
        for i in range(sides + 1):
            angle = i / sides * math.tau
            points.append((math.cos(angle), math.sin(angle), z))
            normals.append((0, 0, sign))
        for i in range(sides):
            tri = (center, center + i + 1, center + i + 2)
            triangles.append(tri if sign > 0 else tuple(reversed(tri)))
    return mesh_node('unit-cylinder', points, normals, triangles)


class ChallengeStudio(ShowBase):
    def __init__(self, run, *, offscreen=False, quality='full', progress=0.0, size=(1600, 1000)):
        loadPrcFileData('', '\n'.join([
            f'win-size {size[0]} {size[1]}', 'window-title Mobile Robotics | 3D Studio',
            'sync-video true', 'show-frame-rate-meter false', 'audio-library-name null',
            'framebuffer-multisample true', 'multisamples 4' if quality == 'full' else 'multisamples 0',
            'textures-power-2 none', 'notify-level warning', 'text-encoding utf8',
        ]))
        super().__init__(windowType='offscreen' if offscreen else None)
        if self.win is None:
            raise RuntimeError('No OpenGL graphics output. Try --quality fast or a display with an OpenGL driver.')
        self.disableMouse()
        self.run_data, self.times = run, [f['t'] for f in run['frames']]
        self.duration = self.times[-1]
        self.clock = self.duration * max(0, min(1, progress))
        self.playing, self.speed, self.offscreen = False, 2.0, offscreen
        self.quality = quality
        self.camera_mode = 'Overview'
        self.focus = Vec3(run['world']['width'] / 2, run['world']['height'] / 2, -.42)
        self.yaw, self.pitch, self.distance = -32.0, 57.0, 19.6
        self.drag = None
        self.current_index = -1
        self.last_ui_time = -1
        self.last_trail_index = -999
        self.slider_lock = False
        self.gui = []
        self.materials = {}
        for font_path in [Path('C:/Windows/Fonts/segoeui.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]:
            if font_path.exists():
                font = self.loader.loadFont(str(Filename.fromOsSpecific(str(font_path))))
                font.setPixelsPerUnit(64)
                TextNode.setDefaultFont(font)
                DirectGuiGlobals.setDefaultFont(font)
                break
        self.box_template, self.cylinder_template = box_mesh(), cylinder_mesh(32 if quality == 'full' else 20)
        self.scene = self.render.attachNewNode('original-course-board')
        self.trail_root = self.render.attachNewNode('recorded-trajectories')
        self.robot = self.render.attachNewNode('robot')
        self.obstacle_nodes = {}
        self.wheels = []
        self.setBackgroundColor(.89, .93, .95)
        self._lighting()
        self._board()
        self._robot_model()
        inset_lens = PerspectiveLens()
        inset_lens.setFov(41)
        inset_lens.setNearFar(.02, 40)
        self.inset_camera = self.makeCamera(self.win, sort=2, scene=self.render,
                                            displayRegion=(.022, .23, .62, .83),
                                            clearDepth=1, clearColor=(.90, .94, .95, 1),
                                            lens=inset_lens, aspectRatio=1.58, camName='robot-detail-camera')
        self.inset_camera.reparentTo(self.render)
        self.inset_region = self.inset_camera.node().getDisplayRegion(0)
        self._build_gui()
        self.accept('window-event', self._resize)
        self.accept('mouse1', self._mouse_down)
        self.accept('mouse1-up', self._mouse_up)
        self.accept('wheel_up', self._zoom, [.87])
        self.accept('wheel_down', self._zoom, [1.15])
        self.accept('space', self.toggle_play)
        self.accept('home', self.reset)
        self.accept('1', self.set_camera_mode, ['Overview'])
        self.accept('2', self.set_camera_mode, ['Robot focus'])
        self.accept('3', self.set_camera_mode, ['Follow'])
        self.accept('escape', self.userExit)
        self._update_scene(force=True)
        self.taskMgr.add(self._tick, 'smooth-recording-playback')

    def material(self, color, shininess=18):
        key = tuple(color) + (shininess,)
        if key not in self.materials:
            material = Material()
            material.setDiffuse(Vec4(*color))
            material.setAmbient(Vec4(color[0] * .62, color[1] * .62, color[2] * .62, 1))
            material.setSpecular(Vec4(.19, .21, .22, 1))
            material.setShininess(shininess)
            self.materials[key] = material
        return self.materials[key]

    def box(self, parent, name, position, scale, color, hpr=None):
        node = parent.attachNewNode(name)
        self.box_template.instanceTo(node)
        node.setPos(*position)
        node.setScale(*scale)
        node.setColor(*color)
        node.setMaterial(self.material(color))
        if hpr:
            node.setHpr(*hpr)
        return node

    def cylinder(self, parent, name, position, radius, depth, color, hpr=None):
        node = parent.attachNewNode(name)
        self.cylinder_template.instanceTo(node)
        node.setPos(*position)
        node.setScale(radius, radius, depth)
        node.setColor(*color)
        node.setMaterial(self.material(color))
        if hpr:
            node.setHpr(*hpr)
        return node

    def stroke(self, points, color=BLACK, width=.045, z=.012, parent=None):
        parent = parent or self.scene
        for start, end in zip(points, points[1:]):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            if length < 1e-8:
                continue
            self.box(parent, 'track-segment', ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, z),
                     (length + width * .2, width, .006), color, (math.degrees(math.atan2(dy, dx)), 0, 0))

    def floor_text(self, text, x, y, *, scale=.1, color=INK, z=.025):
        label = TextNode('floor-label')
        label.setText(str(text))
        label.setTextColor(*color)
        label.setAlign(TextNode.ACenter)
        node = self.scene.attachNewNode(label)
        node.setScale(scale)
        node.setPos(x, y, z)
        node.setP(-90)
        node.setLightOff()
        return node

    def triangle_prism(self, parent, x, y, z, radius, height, color):
        ring = [(radius * math.cos(a), radius * math.sin(a)) for a in (math.pi / 2, math.pi * 7 / 6, math.pi * 11 / 6)]
        points, normals, triangles = [], [], []
        for layer, sign in ((0, -1), (height, 1)):
            start = len(points)
            points.extend([(px, py, layer) for px, py in ring])
            normals.extend([(0, 0, sign)] * 3)
            triangles.append((start, start + 1, start + 2) if sign > 0 else (start, start + 2, start + 1))
        for i in range(3):
            a, b = ring[i], ring[(i + 1) % 3]
            normal = Vec3(b[1] - a[1], a[0] - b[0], 0)
            normal.normalize()
            start = len(points)
            points.extend([(a[0], a[1], 0), (b[0], b[1], 0), (b[0], b[1], height), (a[0], a[1], height)])
            normals.extend([tuple(normal)] * 4)
            triangles.extend([(start, start + 1, start + 2), (start, start + 2, start + 3)])
        node = mesh_node('triangle-packing', points, normals, triangles)
        node.reparentTo(parent)
        node.setPos(x, y, z)
        node.setColor(*color)
        node.setMaterial(self.material(color))
        return node

    def _lighting(self):
        ambient = AmbientLight('cool-room-bounce')
        ambient.setColor((.46, .51, .55, 1))
        self.render.setLight(self.render.attachNewNode(ambient))
        key = DirectionalLight('warm-softbox')
        key.setColor((.86, .83, .76, 1))
        if self.quality == 'full':
            key.setShadowCaster(True, 2048, 2048)
            key.getLens().setFilmSize(13, 13)
            key.getLens().setNearFar(.5, 25)
        node = self.render.attachNewNode(key)
        node.setPos(-1, -2, 10)
        node.lookAt(3.5, 3.5, 0)
        self.render.setLight(node)
        fill = DirectionalLight('cool-window-fill')
        fill.setColor((.17, .22, .27, 1))
        node = self.render.attachNewNode(fill)
        node.setHpr(-100, -24, 0)
        self.render.setLight(node)
        self.render.setShaderAuto()
        fog = Fog('distant-studio-haze')
        fog.setColor(.89, .93, .95)
        fog.setLinearRange(17, 42)
        self.render.setFog(fog)

    def _board(self):
        run, track = self.run_data, self.run_data['track']
        width, height = run['world']['width'], run['world']['height']
        self.box(self.scene, 'studio-floor', (3.5, 3.5, -.22), (70, 70, .2), (.81, .86, .88, 1))
        self.box(self.scene, 'map-board-edge', (width / 2, height / 2, -.078), (width + .2, height + .2, .15), (.24, .32, .35, 1))
        self.box(self.scene, 'course-map-board', (width / 2, height / 2, -.015), (width + .08, height + .08, .04), WHITE)
        # Subtle measurement grid, separate from the black navigation lines.
        for n in range(int(width) + 1):
            self.stroke([(n, 0), (n, height)], (.90, .92, .89, 1), .006, .007)
        for n in range(int(height) + 1):
            self.stroke([(0, n), (width, n)], (.90, .92, .89, 1), .006, .007)
        for polyline in track['polylines']:
            points = polyline['points']
            if polyline.get('kind') == 'black_line':
                self.stroke(points)
            elif polyline.get('kind') == 'unmarked':
                self.stroke(points, (.57, .65, .66, 1), .014, .012)
        color_map = {'Red': RED, 'Blue': (.15, .53, .72, 1), 'Yellow': (.89, .70, .15, 1), 'Green': (.13, .55, .34, 1)}
        for node in track.get('nodes', []):
            self.cylinder(self.scene, 'graph-node', (node['x'], node['y'], .025), .075, .018, color_map.get(node.get('color'), TEAL))
            self.floor_text(node['id'], node['x'] + .12, node['y'] + .06, scale=.13)
        for marker in track.get('markers', []):
            self.box(self.scene, 'task-floor-marker', (marker['x'], marker['y'], .018), (.09, .12, .009), RED)
            self.floor_text(f"{marker['task']:02d}", marker['x'] + .18, marker['y'] - .025, scale=.105, color=RED)
        for marker in track.get('blue_markers', []):
            self.box(self.scene, 'blue-floor-marker', (marker['x'], marker['y'], .019), (.085, .12, .008), (.08, .42, .67, 1))
        for item in track.get('containers', []):
            box = self.box(self.scene, 'cargo-bin', (item['x'], item['y'], .09), (.25, .22, .17), (.52, .65, .69, 1))
            self.box(box, 'bin-rim', (0, 0, .53), (.97, .97, .08), (.23, .38, .42, 1))
            self.floor_text(item['id'], item['x'] + .02, item['y'] - .23, scale=.07, color=MUTED)
        for item in track.get('packaging', []):
            x, y = item['x'], item['y']
            self.box(self.scene, 'packing-platform', (x, y, .045), (.29, .28, .075), (.73, .62, .40, 1))
            if item['shape'] == 'Circle':
                self.cylinder(self.scene, 'circular-packing', (x, y, .12), .085, .1, (.82, .72, .51, 1))
            else:
                if item['shape'] == 'Triangle':
                    self.triangle_prism(self.scene, x, y, .085, .10, .08, (.82, .72, .51, 1))
                else:
                    self.box(self.scene, 'packing-shape', (x, y, .115), (.16, .12, .08), (.82, .72, .51, 1))
            self.floor_text(item['shape'].upper(), x - .16, y - .23, scale=.075, color=MUTED)
        charger = track.get('charger', {})
        if charger:
            x, y = charger['x'], charger['y']
            self.box(self.scene, 'charge-pad', (x, y, .023), (.45, .36, .035), (.14, .35, .39, 1))
            for dx in (-.1, .1):
                self.box(self.scene, 'charge-contact', (x + dx, y, .047), (.038, .19, .008), (.86, .71, .32, 1))
            self.floor_text('CHARGE', x, y - .34, scale=.10, color=TEAL)
        # The scene has the same dynamic obstacles as the recorded run.
        for obstacle in run['world'].get('obstacles', []):
            root = self.scene.attachNewNode('dynamic-obstacle')
            x, y, w, h = obstacle['x'], obstacle['y'], obstacle['w'], obstacle['h']
            self.box(root, 'obstacle-body', (x + w / 2, y + h / 2, .12), (w, h, .24), (.74, .40, .18, 1))
            for dx in (-w * .22, w * .22):
                self.box(root, 'obstacle-reflector', (x + w / 2 + dx, y - .004, .15), (.025, .008, .1), (.96, .86, .52, 1))
            root.hide()
            self.obstacle_nodes[obstacle['id']] = root
        # A single traffic-light reference beside the original entry point.
        self.cylinder(self.scene, 'signal-pole', (.16, .5, .28), .014, .54, (.24, .29, .31, 1))
        self.box(self.scene, 'signal-housing', (.16, .5, .57), (.09, .065, .20), BLACK)
        self.signal_lamps = []
        for z, color in ((.63, RED), (.57, (.92, .65, .14, 1)), (.51, (.12, .63, .35, 1))):
            self.signal_lamps.append(self.cylinder(self.scene, 'signal-lens', (.16, .462, z), .023, .012, color, (0, 90, 0)))
        self.floor_text('ORIGINAL COURSE CHALLENGE', width / 2, -.38, scale=.14, color=MUTED)

    def _robot_model(self):
        # +Y is model-forward; pose theta=0 means map +X, handled in _update_scene.
        r = self.robot.attachNewNode('procedural-brickpi-cart')
        r.setScale(1.7)
        gray, light_gray = (.43, .47, .48, 1), (.77, .79, .77, 1)
        self.box(r, 'lower-chassis', (0, -.015, .108), (.26, .31, .035), gray)
        self.box(r, 'EV3-white-housing', (0, .0, .21), (.19, .18, .12), (.89, .90, .86, 1))
        self.box(r, 'EV3-top-bezel', (0, .0, .274), (.16, .15, .016), (.69, .72, .70, 1))
        self.box(r, 'LCD-glass', (0, .025, .284), (.12, .075, .006), (.21, .36, .39, 1))
        for n, length in enumerate((.076, .046, .068)):
            self.box(r, 'LCD-status-line', (-.01, .047 - n * .018, .288), (length, .005, .002), (.55, .78, .72, 1))
        self.cylinder(r, 'EV3-control-pad', (0, -.05, .287), .022, .007, (.25, .28, .28, 1))
        self.cylinder(r, 'EV3-control-center', (0, -.05, .293), .010, .004, (.65, .70, .67, 1))
        for side in (-1, 1):
            self.box(r, 'drive-motor-white', (side * .12, -.04, .15), (.07, .16, .09), (.83, .85, .82, 1))
            self.box(r, 'drive-motor-red-cap', (side * .13, -.105, .15), (.075, .035, .07), RED)
            axle = r.attachNewNode('animated-wheel')
            axle.setPos(side * .174, -.065, .094)
            self.cylinder(axle, 'rubber-tire', (0, 0, 0), .094, .055, (.055, .062, .065, 1), (0, 0, 90))
            for i in range(28 if self.quality == 'full' else 18):
                angle = math.tau * i / (28 if self.quality == 'full' else 18)
                tread = self.box(axle, 'tire-tread', (0, math.cos(angle) * .091, math.sin(angle) * .091), (.059, .016, .012), (.09, .10, .10, 1))
                tread.setP(math.degrees(angle))
            self.cylinder(axle, 'wheel-gray-hub', (side * .031, 0, 0), .052, .012, light_gray, (0, 0, 90))
            self.cylinder(axle, 'wheel-hub-dark-inset', (side * .039, 0, 0), .028, .015, gray, (0, 0, 90))
            for i in range(6):
                angle = i * math.tau / 6
                self.cylinder(axle, 'hub-bolt', (side * .049, math.cos(angle) * .038, math.sin(angle) * .038), .0046, .003, (.20, .23, .23, 1), (0, 0, 90))
            self.wheels.append(axle)
        # Rear castor and exposed green single-board computer.
        self.cylinder(r, 'rear-castor', (0, -.16, .055), .038, .035, (.15, .18, .18, 1), (0, 0, 90))
        self.box(r, 'black-BrickPi-PCB', (0, -.125, .187), (.15, .095, .012), (.075, .10, .10, 1))
        for x in (-.064, .064):
            for y in (-.16, -.09):
                self.cylinder(r, 'board-standoff', (x, y, .201), .007, .025, (.63, .64, .57, 1))
        self.box(r, 'processor-package', (-.02, -.12, .2), (.045, .039, .012), BLACK)
        self.box(r, 'USB-stack', (.063, -.14, .208), (.035, .047, .026), (.61, .65, .65, 1))
        for i in range(10):
            self.box(r, 'GPIO-pin', (-.059 + i * .012, -.165, .207), (.005, .008, .012), (.79, .66, .31, 1))
        # Perforated Technic-style mast, crossmembers and pallet forks.
        for x in (-.105, .105):
            self.box(r, 'fork-upright', (x, .11, .205), (.027, .035, .31), light_gray)
            for z in (.09, .14, .19, .24, .29, .34):
                self.cylinder(r, 'Technic-pin-hole', (x, .13, z), .008, .003, (.17, .20, .20, 1), (0, 90, 0))
            self.box(r, 'fork-base', (x, .21, .048), (.025, .31, .025), gray)
            for y in (.13, .18, .23, .28, .33):
                self.cylinder(r, 'fork-pin-hole', (x, y, .063), .006, .003, (.13, .16, .17, 1))
        self.box(r, 'upper-crossmember', (0, .11, .345), (.23, .04, .027), gray)
        self.box(r, 'camera-mast', (.034, .035, .392), (.033, .04, .245), light_gray)
        self.box(r, 'camera-red-bracket', (.034, .068, .492), (.105, .038, .073), RED)
        self.box(r, 'camera-module', (.034, .092, .499), (.069, .021, .043), (.18, .19, .20, 1))
        self.cylinder(r, 'camera-lens', (.034, .108, .499), .018, .016, BLACK, (0, 90, 0))
        self.cylinder(r, 'camera-lens-glass', (.034, .117, .499), .011, .003, (.13, .29, .36, 1), (0, 90, 0))
        # Ultrasonic pair below the camera mast.
        self.box(r, 'ultrasonic-pod', (0, .14, .297), (.13, .044, .065), (.29, .32, .33, 1))
        for x in (-.039, .039):
            self.cylinder(r, 'ultrasonic-rim', (x, .166, .297), .024, .014, RED, (0, 90, 0))
            self.cylinder(r, 'ultrasonic-face', (x, .177, .297), .0175, .009, (.13, .16, .18, 1), (0, 90, 0))
        # Newly modelled camera ribbon and control cable, not copied assets.
        cables = [
            [(.034, .04, .50), (.07, -.01, .56), (.08, -.12, .55), (.02, -.15, .25)],
            [(-.04, -.14, .21), (-.12, -.1, .28), (-.13, .025, .31), (-.07, .11, .30)],
        ]
        for points in cables:
            lines = LineSegs('camera-cable')
            lines.setThickness(3.5)
            lines.setColor(.08, .09, .10, 1)
            lines.moveTo(*points[0])
            for point in points[1:]:
                lines.drawTo(*point)
            r.attachNewNode(lines.create())
        self.cargo = self.box(r, 'carried-container', (0, .235, .113), (.15, .16, .105), (.52, .66, .68, 1))
        self.cargo.hide()
        # A soft contact shadow complements the engine's directional shadows.
        image = PNMImage(96, 96, 4)
        for x in range(96):
            for y in range(96):
                d = ((x - 47.5) / 39) ** 2 + ((y - 47.5) / 39) ** 2
                image.setXelA(x, y, 0, 0, 0, max(0, math.exp(-d * 2.3) * .20 - .003))
        texture = Texture('soft-contact-shadow')
        texture.load(image)
        card = CardMaker('contact-shadow')
        card.setFrame(-.37, .37, -.35, .42)
        shadow = self.robot.attachNewNode(card.generate())
        shadow.setP(-90)
        shadow.setZ(.012)
        shadow.setTexture(texture)
        shadow.setTransparency(TransparencyAttrib.MAlpha)
        shadow.setDepthWrite(False)
        shadow.setLightOff()
        shadow.setShaderOff()

    def _build_gui(self):
        self._last_window_size = (self.win.getXSize(), self.win.getYSize())
        for widget in self.gui:
            widget.destroy()
        self.gui = []
        self.aspect = self.getAspectRatio()
        right = self.aspect * .52
        self.main_region = self.camNode.getDisplayRegion(0)
        self.main_region.setDimensions(0, .76, 0, 1)
        self.camLens.setAspectRatio(self.aspect * .76)
        self.camLens.setFov(42)
        self.camLens.setNearFar(.025, 95)

        def frame(**kwargs):
            item = DirectFrame(**kwargs)
            self.gui.append(item)
            return item

        def label(text, pos, scale=.033, color=INK, **kwargs):
            item = DirectLabel(text=text, text_scale=scale, text_fg=color,
                               frameColor=(0, 0, 0, 0), pos=pos, text_align=TextNode.ALeft, **kwargs)
            self.gui.append(item)
            return item

        def button(text, x, y, width, command, accent=False):
            item = DirectButton(text=text, text_scale=.030, text_fg=(1, 1, 1, 1) if accent else INK,
                                frameColor=TEAL if accent else (.90, .94, .94, 1),
                                frameSize=(-width / 2, width / 2, -.039, .039),
                                pos=(x, 0, y), relief=1, command=command, text_pos=(0, -.009))
            self.gui.append(item)
            return item

        frame(frameColor=(.965, .978, .974, .98), frameSize=(right, self.aspect, -.985, .985))
        frame(frameColor=(.89, .93, .93, 1), frameSize=(right, right + .004, -.985, .985))
        label('MOBILE ROBOTICS / 3D STUDIO', (-self.aspect + .10, 0, .906), .026, TEAL)
        label('Original Challenge', (-self.aspect + .10, 0, .824), .058)
        label('Physical team practical: Sep 2025   /   Simulation reconstruction: Oct 2026',
              (-self.aspect + .10, 0, .766), .024, MUTED)
        label('TEAL  Reference trajectory     BLUE  Encoder odometry',
              (-self.aspect + .10, 0, .712), .023, MUTED)
        self.inset_caption = label('ROBOT DETAIL / 3D CAMERA', (-self.aspect + .10, 0, .682), .023, TEAL)
        panel_x = right + .06
        label('COURSE WORKFLOW', (panel_x, 0, .894), .026, TEAL)
        label('Eight connected tasks', (panel_x, 0, .832), .039)
        self.state_label = label('READY', (panel_x, 0, .758), .028, MUTED)
        self.task_widgets = []
        for i, title in enumerate(TASK_LABELS):
            row = .658 - i * .101
            task_number = label(f'{i + 1:02d}', (panel_x, 0, row), .031, MUTED)
            task_title = label(title, (panel_x + .075, 0, row), .029, MUTED)
            self.task_widgets.append((task_number, task_title))
        frame(frameColor=(.85, .90, .89, 1), frameSize=(panel_x, self.aspect - .05, -.20, -.197))
        label('RECORDING', (panel_x, 0, -.26), .023, TEAL)
        self.time_label = label('', (panel_x, 0, -.322), .038)
        self.distance_label = label('', (panel_x, 0, -.383), .028, MUTED)
        self.cargo_label = label('', (panel_x, 0, -.44), .028, MUTED)
        label('CAMERA', (panel_x, 0, -.542), .023, TEAL)
        for i, mode in enumerate(('Overview', 'Robot focus', 'Follow')):
            button(mode, panel_x + .30, -.613 - i * .088, .59, lambda mode=mode: self.set_camera_mode(mode))
        label('Drag: orbit    Wheel: zoom', (panel_x, 0, -.91), .023, MUTED)
        label('Space: play / pause    Home: reset', (panel_x, 0, -.95), .021, MUTED)
        left = -self.aspect + .12
        bottom_width = right - left - .11
        frame(frameColor=(.96, .978, .974, .96), frameSize=(left - .05, right - .055, -.972, -.775))
        self.play_button = button('PLAY', left + .12, -.824, .24, self.toggle_play, True)
        button('Reset', left + .40, -.824, .24, self.reset)
        self.speed_button = button(f'{self.speed:g}x', left + .68, -.824, .22, self.cycle_speed)
        self.camera_label = label(self.camera_mode.upper(), (left + .87, 0, -.835), .023, MUTED)
        self.seek = DirectSlider(range=(0, self.duration), value=self.clock, pageSize=max(1, self.duration / 60),
                                 frameColor=(.76, .84, .84, 1), thumb_frameColor=TEAL,
                                 frameSize=(-1, 1, -.011, .011), thumb_frameSize=(-.017, .017, -.035, .035),
                                 thumb_relief=0,
                                 scale=(bottom_width / 2, 1, .52), pos=(left + bottom_width / 2, 0, -.905),
                                 command=self._seek_changed)
        self.gui.append(self.seek)
        # Explicit marker is more readable than the engine's platform-specific
        # slider thumb style; the slider still provides the complete seek input.
        self.seek_indicator = frame(frameColor=TEAL, frameSize=(-.008, .008, -.017, .017), pos=(left, 0, -.905))
        self.seek_bounds = (left, left + bottom_width)
        self.last_ui_time = -1

    def _resize(self, window):
        # Focus/expose/property notifications are also "window-event" events.
        # Rebuild widgets only for a changed drawable size, never every event.
        size = (window.getXSize(), window.getYSize()) if window else None
        if window is self.win and hasattr(self, 'gui') and size != getattr(self, '_last_window_size', None):
            self._build_gui()

    def toggle_play(self):
        if self.clock >= self.duration:
            self.clock = 0
        self.playing = not self.playing
        self.play_button['text'] = 'PAUSE' if self.playing else 'PLAY'

    def reset(self):
        self.clock, self.playing = 0.0, False
        self.play_button['text'] = 'PLAY'
        self._update_scene(force=True)

    def cycle_speed(self):
        speeds = [.5, 1, 2, 4, 8]
        self.speed = speeds[(speeds.index(self.speed) + 1) % len(speeds)]
        self.speed_button['text'] = f'{self.speed:g}x'

    def _seek_changed(self):
        if self.slider_lock or not hasattr(self, 'seek'):
            return
        # Panda queues ADJUST events, including those emitted by programmatic
        # value changes. They arrive after slider_lock has been released.
        # An event already matching the current playback clock is an echo,
        # not user input; consuming it avoids re-queuing ADJUST indefinitely.
        value = float(self.seek.getValue())
        # PGSliderBar may round through a native float; tolerate sub-millisecond
        # error even late in a long recording, without swallowing human seeks.
        if abs(value - self.clock) <= max(1e-5, self.duration * 1e-6):
            return
        self.clock = value
        self.playing = False
        if hasattr(self, 'play_button'):
            self.play_button['text'] = 'PLAY'
        self._update_scene(force=True)

    def set_camera_mode(self, mode):
        self.camera_mode = mode
        if mode == 'Overview':
            self.focus = Vec3(self.run_data['world']['width'] / 2, self.run_data['world']['height'] / 2, -.42)
            self.yaw, self.pitch, self.distance = -32, 57, 19.6
        elif mode == 'Robot focus':
            self.distance, self.pitch = 2.6, 33
        else:
            self.distance, self.pitch = 3.2, 34
        self.camera_label['text'] = mode.upper()
        self.inset_region.setActive(mode == 'Overview')
        self.inset_caption.show() if mode == 'Overview' else self.inset_caption.hide()
        self._position_camera(snap=True)

    def _mouse_down(self):
        if self.mouseWatcherNode.hasMouse():
            mouse = self.mouseWatcherNode.getMouse()
            if mouse.x < .52 and mouse.y > -.74:
                self.drag = (mouse.x, mouse.y)

    def _mouse_up(self):
        self.drag = None

    def _zoom(self, factor):
        if self.mouseWatcherNode.hasMouse() and self.mouseWatcherNode.getMouse().x > .52:
            return
        self.distance = max(1.2, min(24, self.distance * factor))

    def _position_camera(self, snap=False):
        current, _ = interpolated_frame(self.run_data, self.times, self.clock)
        pose = current['truth']
        if self.camera_mode != 'Overview':
            target = Vec3(pose[0], pose[1], .28)
            self.focus = target if snap else self.focus * .86 + target * .14
        if self.camera_mode == 'Follow':
            desired = math.degrees(pose[2]) - 90
            delta = (desired - self.yaw + 180) % 360 - 180
            self.yaw = desired if snap else self.yaw + delta * .055
        yaw, pitch = math.radians(self.yaw), math.radians(self.pitch)
        horizontal = self.distance * math.cos(pitch)
        self.camera.setPos(self.focus.x + math.sin(yaw) * horizontal,
                           self.focus.y - math.cos(yaw) * horizontal,
                           self.focus.z + self.distance * math.sin(pitch))
        self.camera.lookAt(self.focus)
        # The extra viewport is a real perspective camera into the same scene.
        # It follows the recorded robot without a second physics simulation.
        theta = pose[2]
        self.inset_camera.setPos(pose[0] + math.cos(theta) * 1.7 - math.sin(theta) * 1.15,
                                 pose[1] + math.sin(theta) * 1.7 + math.cos(theta) * 1.15, 1.24)
        self.inset_camera.lookAt(pose[0], pose[1], .39)

    def _draw_trails(self, index):
        self.trail_root.getChildren().detach()
        for key, color in [('truth', TEAL), ('odom', BLUE)]:
            lines = LineSegs(key + '-trajectory')
            lines.setThickness(2.4 if key == 'truth' else 1.3)
            lines.setColor(*color)
            samples = self.run_data['frames'][:index + 1: max(1, index // 500)]
            if not samples:
                continue
            lines.moveTo(samples[0][key][0], samples[0][key][1], .029)
            for frame in samples[1:]:
                lines.drawTo(frame[key][0], frame[key][1], .029)
            node = self.trail_root.attachNewNode(lines.create())
            node.setLightOff()
        self.last_trail_index = index

    def _update_scene(self, force=False):
        frame, index = interpolated_frame(self.run_data, self.times, self.clock)
        pose = frame['truth']
        self.robot.setPos(pose[0], pose[1], .018)
        self.robot.setH(math.degrees(pose[2]) - 90)
        traveled = frame.get('metrics', {}).get('distance', 0)
        for wheel in self.wheels:
            wheel.setP(-math.degrees(traveled / .094))
        self.cargo.show() if frame.get('loaded') else self.cargo.hide()
        active = set(frame.get('active_obstacles', []))
        for key, node in self.obstacle_nodes.items():
            node.show() if key in active else node.hide()
        if force or index < self.last_trail_index or abs(index - self.last_trail_index) > 7:
            self._draw_trails(index)
        self._position_camera(snap=force)
        if force or abs(self.clock - self.last_ui_time) > .12:
            task = max(1, min(8, int(frame.get('task', 1))))
            self.state_label['text'] = f'{TASK_SHORT[task - 1]}  /  TASK {task:02d}'
            for i, widgets in enumerate(self.task_widgets):
                for item in widgets:
                    item['text_fg'] = TEAL if i + 1 == task else INK if i + 1 < task else MUTED
            self.time_label['text'] = f'{self.clock:05.1f} / {self.duration:.1f} s'
            self.distance_label['text'] = f'Travelled    {traveled:.2f} m'
            self.cargo_label['text'] = 'Payload      LOADED' if frame.get('loaded') else 'Payload      EMPTY'
            self.slider_lock = True
            self.seek['value'] = self.clock
            low, high = self.seek_bounds
            self.seek_indicator.setX(low + (high - low) * self.clock / max(.001, self.duration))
            self.slider_lock = False
            self.last_ui_time = self.clock
        self.current_index = index

    def _tick(self, task):
        dt = min(.1, globalClock.getDt())
        if self.playing:
            self.clock = min(self.duration, self.clock + dt * self.speed)
            if self.clock >= self.duration:
                self.playing = False
                self.play_button['text'] = 'PLAY'
        if self.drag is not None and self.mouseWatcherNode.hasMouse():
            mouse = self.mouseWatcherNode.getMouse()
            dx, dy = mouse.x - self.drag[0], mouse.y - self.drag[1]
            self.yaw -= dx * 130
            self.pitch = max(12, min(82, self.pitch + dy * 100))
            self.drag = (mouse.x, mouse.y)
        self._update_scene()
        return task.cont

    def save_frame(self, path, *, progress=None, camera_mode=None):
        if progress is not None:
            self.clock = self.duration * max(0, min(1, progress))
        if camera_mode:
            self.set_camera_mode(camera_mode)
        self._update_scene(force=True)
        # Let auto shaders and shadow buffers settle before capturing.
        for _ in range(6):
            self.graphicsEngine.renderFrame()
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = self.win.saveScreenshot(Filename.fromOsSpecific(str(path)))
        if not ok:
            raise RuntimeError(f'Could not save screenshot: {path}')
        return path


def run_studio(data=None, render_frame=None, progress=.5, offscreen=False, quality='full', camera='Overview'):
    run = load_recording(data)
    app = ChallengeStudio(run, offscreen=bool(offscreen or render_frame), quality=quality,
                          progress=progress if render_frame else 0)
    if render_frame:
        path = app.save_frame(render_frame, progress=progress, camera_mode=camera)
        app.destroy()
        print(f'Rendered real 3D scene: {path}')
        return path
    app.run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--render-frame', type=Path)
    parser.add_argument('--progress', type=float, default=.5)
    parser.add_argument('--offscreen', action='store_true')
    parser.add_argument('--quality', choices=('full', 'fast'), default='full')
    parser.add_argument('--camera', choices=('overview', 'robot', 'follow'), default='overview')
    args = parser.parse_args()
    run_studio(args.data, args.render_frame, args.progress, args.offscreen, args.quality,
               {'overview': 'Overview', 'robot': 'Robot focus', 'follow': 'Follow'}[args.camera])


if __name__ == '__main__':
    main()
