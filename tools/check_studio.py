"""Optional OpenGL integration check. Requires requirements-3d.txt and a graphics driver."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from native_3d import ChallengeStudio, load_recording
from panda3d.core import ClockObject

run=load_recording()
app=ChallengeStudio(run,offscreen=True,quality='fast',size=(1000,700))
try:
    identity=int(app.robot.node().this)
    app.playing=True
    clock=ClockObject.getGlobalClock()
    clock.setMode(ClockObject.MNonRealTime)
    clock.setDt(1/60)
    for _ in range(12):app.taskMgr.step()
    assert app.clock>0 and app.playing, 'Playback stopped during automatic slider updates'
    for task in range(1,9):
        sample=next(frame for frame in run['frames'] if int(frame['task'])==task)
        app.clock=sample['t']+.01
        app._update_scene(force=True)
        app.graphicsEngine.renderFrame()
        assert int(app.robot.node().this)==identity, 'Robot scene was recreated at a task boundary'
        assert app.playing, 'A task boundary paused playback'
    for mode in ('Robot focus','Follow','Overview'):
        app.set_camera_mode(mode)
        app.graphicsEngine.renderFrame()
    app.reset()
    assert app.clock==0 and not app.playing
    print(json.dumps({'continuous_scene_across_eight_tasks':True,'automatic_slider_keeps_playing':True,
                      'camera_modes':3,'reset':True,'driver':app.win.getGsg().getDriverRenderer()},indent=2))
finally:
    app.destroy()
