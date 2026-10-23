#!/usr/bin/env python3
"""Clone the repository, install requirements, then run: python main.py."""
import argparse
import sys
from pathlib import Path
from native_view import DesktopApp, load_run, render_presentation

def main():
    parser=argparse.ArgumentParser(description="Native original-layout intralogistics challenge simulator")
    parser.add_argument("--data",type=Path,help="Use a different original challenge JSON recording")
    parser.add_argument("--render-frame",type=Path,help="Save a presentation PNG without opening a window")
    parser.add_argument("--progress",type=float,default=.5,help="Fraction of the recording for --render-frame, 0..1")
    parser.add_argument("--view",choices=["3d","topdown"],default="3d")
    parser.add_argument("--studio",action="store_true",help="Open the optional real-time OpenGL 3D Studio")
    parser.add_argument("--regenerate",action="store_true",help="Recompute the simulation from editable config before opening/exporting")
    parser.add_argument("--storage",choices=["g","i","k","l"],default="i",help="Warehouse storage target when regenerating")
    parser.add_argument("--smoke-test",action="store_true",help="Validate original-layout data and native renderer without opening a window")
    parser.add_argument("--gui-smoke-test",action="store_true",help="Open a native window, draw once and close after 1 second")
    args=parser.parse_args()
    if args.studio:
        if args.regenerate:load_run(args.data,regenerate=True,storage=args.storage)
        runtime=Path(__file__).resolve().parent/".runtime"
        if runtime.exists():sys.path.insert(0,str(runtime))
        try:
            from native_3d import run_studio
        except ImportError as exc:
            parser.error(f"3D Studio requires Panda3D. Install: python -m pip install 'panda3d>=1.10.15,<1.11'. Details: {exc}")
        run_studio(data=args.data,render_frame=args.render_frame,progress=args.progress,offscreen=bool(args.render_frame))
        return
    run=load_run(args.data,regenerate=args.regenerate,storage=args.storage)
    index=round(max(0,min(1,args.progress))*(len(run["frames"])-1))
    if args.render_frame:
        args.render_frame.parent.mkdir(parents=True,exist_ok=True)
        render_presentation(run,index,view=args.view).save(args.render_frame)
        print(f"Rendered {args.render_frame}")
    if args.smoke_test:
        assert len(run.get("tasks",[]))==8,"Expected actual eight original challenge tasks"
        assert {int(f["task"]) for f in run["frames"]}==set(range(1,9)),"All eight tasks must execute"
        for view in ("3d","topdown"):
            frame=render_presentation(run,index,view=view)
            assert frame.size==(1360,900)
        print(f"PASS native render/data: {len(run['frames'])} frames, 8 tasks, 2 views")
    if not(args.render_frame or args.smoke_test) or args.gui_smoke_test:
        app=DesktopApp(run)
        if args.gui_smoke_test:app.root.after(1000,app.root.destroy)
        app.mainloop()

if __name__=="__main__":main()
