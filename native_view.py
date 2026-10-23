"""Native desktop viewer and Pillow renderer for the reconstructed course board.

This draws the actual eight-task layout from editable geometry, not a generic
warehouse. It visualizes recorded planar simulation; it is not Gazebo physics.
"""
from __future__ import annotations
import json
import math
import sys
import time
import bisect
import subprocess
from functools import lru_cache
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageTk

ROOT = Path(__file__).resolve().parent
INK = "#183141"
BLUE = "#3168c7"
TEAL = "#078c8c"
PAPER = "#f1f3ed"
MUTED = "#71838b"
TASKS_EN = ["Green-light start", "Barcode match & pickup", "Dijkstra storage & exit", "MQTT worker handover", "Follow worker QR", "Wait & bypass obstacle", "Select packaging station", "Return to charging"]

def english_state(frame):
    return frame.get("state_en") or TASKS_EN[max(0,min(7,int(frame.get("task",1))-1))]

def interpolated_frame(run, at_time):
    """Interpolate recorded poses without interpolating across the yaw wrap."""
    frames=run["frames"]
    times=run.get("_viewer_times")
    if times is None:
        times=[f["t"] for f in frames]
        run["_viewer_times"]=times
    index=max(0,min(len(frames)-1,bisect.bisect_right(times,at_time)-1))
    first=frames[index]
    if index==len(frames)-1:return index,first
    second=frames[index+1]
    alpha=max(0,min(1,(at_time-first["t"])/max(1e-9,second["t"]-first["t"])))
    result=dict(first);result["t"]=first["t"]+alpha*(second["t"]-first["t"])
    for key in ("truth","odom"):
        a,b=first[key],second[key]
        angle=math.atan2(math.sin(b[2]-a[2]),math.cos(b[2]-a[2]))
        result[key]=[a[0]+alpha*(b[0]-a[0]),a[1]+alpha*(b[1]-a[1]),a[2]+alpha*angle]
    result["metrics"]=dict(first.get("metrics",{}))
    for key,value in first.get("metrics",{}).items():
        if isinstance(value,(int,float)) and isinstance(second.get("metrics",{}).get(key),(int,float)):
            result["metrics"][key]=value+alpha*(second["metrics"][key]-value)
    return index,result

@lru_cache(maxsize=128)
def font(size=16, bold=False):
    candidates = [Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()

def load_run(path=None, regenerate=False, storage="i"):
    path = Path(path) if path else ROOT / "docs/data/original-challenge.json"
    if regenerate or not path.exists():
        sys.path.insert(0,str(ROOT/"src"))
        from intralogistics_sim.original_challenge import simulate_original_challenge
        data=simulate_original_challenge(storage=storage,seed=42)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("frames") or not data.get("track", {}).get("polylines"):
        raise ValueError("Original challenge data must include frames and editable track polylines")
    return data

class SceneRenderer:
    def __init__(self, run):
        self.run = run
        self._static_cache = {}

    def render(self, index=0, size=(980,740), view="3d", trails=True, sensors=True, sample_frame=None):
        width, height = size
        image = Image.new("RGB",size,PAPER)
        draw = ImageDraw.Draw(image)
        world=self.run["world"]
        frames=self.run["frames"]; index=max(0,min(index,len(frames)-1)); f=sample_frame or frames[index]
        ww,wh=world["width"],world["height"]
        if view=="topdown":
            raw=lambda x,y,z=0:(x,-y-z*.35)
        else:
            raw=lambda x,y,z=0:(x+.37*y,.22*x-.71*y-.92*z)
        corners=[raw(x,y) for x,y in ((0,0),(ww,0),(ww,wh),(0,wh))]
        minx,maxx=min(p[0] for p in corners),max(p[0] for p in corners)
        miny,maxy=min(p[1] for p in corners),max(p[1] for p in corners)
        margin=45
        scale=min((width-2*margin)/(maxx-minx),(height-2*margin)/(maxy-miny))
        ox=(width-(maxx-minx)*scale)/2-minx*scale
        oy=(height-(maxy-miny)*scale)/2-miny*scale
        def p(x,y,z=0):
            u,v=raw(x,y,z)
            return(ox+u*scale,oy+v*scale)
        def cube(x,y,w=.22,d=.2,h=.16,color="#bdcbd3"):
            a,b,c,d0=[p(px,py,0) for px,py in ((x-w/2,y-d/2),(x+w/2,y-d/2),(x+w/2,y+d/2),(x-w/2,y+d/2))]
            at,bt,ct,dt=[p(px,py,h) for px,py in ((x-w/2,y-d/2),(x+w/2,y-d/2),(x+w/2,y+d/2),(x-w/2,y+d/2))]
            draw.polygon([a,b,bt,at],fill="#91a8b4",outline="#617c8a")
            draw.polygon([b,c,ct,bt],fill="#7e99a9",outline="#617c8a")
            draw.polygon([at,bt,ct,dt],fill=color,outline="#6e8791")
        cache_key=(size,view)
        if len(self._static_cache)>3:self._static_cache.clear()
        if cache_key not in self._static_cache:
            plane=[p(x,y) for x,y in ((0,0),(ww,0),(ww,wh),(0,wh))]
            if view!="topdown":
                draw.polygon([(x+3,y+13) for x,y in plane],fill="#d0dcd9")
            draw.polygon(plane,fill="#fffef9",outline="#bccfcd",width=2)
            # A subtle measured reference, beneath the original black floor tracks.
            for n in range(int(ww)+1): draw.line([p(n,0),p(n,wh)],fill="#ecf0ea",width=1)
            for n in range(int(wh)+1): draw.line([p(0,n),p(ww,n)],fill="#ecf0ea",width=1)
            track=self.run["track"]
            for path in track.get("polylines",[]):
                pts=[p(*xy) for xy in path["points"]]
                if len(pts)<2: continue
                kind=path.get("kind","black_line")
                if kind=="black_line": draw.line(pts,fill="#202d32",width=max(3,round(scale*.055)),joint="curve")
                elif kind=="unmarked":
                    for a,b in zip(pts,pts[1:]):
                        dist=math.hypot(b[0]-a[0],b[1]-a[1])
                        for j in range(max(1,int(dist/12))):
                            frac=j*12/max(dist,1);g=min(1,frac+5/max(dist,1))
                            draw.line([(a[0]+(b[0]-a[0])*frac,a[1]+(b[1]-a[1])*frac),(a[0]+(b[0]-a[0])*g,a[1]+(b[1]-a[1])*g)],fill="#aebbbb",width=2)
                else: draw.line(pts,fill="#b8bfc5",width=2)
            for item in track.get("containers",[]):
                x,y=item.get("x",0),item.get("y",0)
                cube(x,y,item.get("w",.3),item.get("h",.24),.16)
            color_map={"red":"#e44848","blue":"#25b9db","green":"#13864d","yellow":"#f0c74a","Black":"#202d32","Red":"#e44848","Blue":"#25b9db","Yellow":"#f0c74a","Green":"#13864d"}
            for node in track.get("nodes",[]):
                x,y=p(node["x"],node["y"])
                col=color_map.get(node.get("color"),node.get("color","#30b7d1"))
                r=max(3,scale*.052)
                draw.rounded_rectangle((x-r,y-r,x+r,y+r),radius=1,fill=col)
                if node.get("id"):
                    label=str(node["id"])
                    draw.text((x+7,y-10),label,fill="#587078",font=font(max(10,round(width/90))))
            for marker in track.get("markers",[]):
                x,y=p(marker["x"],marker["y"])
                draw.rectangle((x-4,y-5,x+4,y+5),fill=color_map.get(marker.get("color"),"#e44848"))
                r=max(10,width/95)
                label=str(marker.get("task",""))
                labelxy=(x+17,y+1)
                draw.ellipse((labelxy[0]-r,labelxy[1]-r,labelxy[0]+r,labelxy[1]+r),fill="#fffef9",outline="#809998",width=1)
                bbox=draw.textbbox((0,0),label,font=font(12,True))
                draw.text((labelxy[0]-(bbox[2]-bbox[0])/2,labelxy[1]-9),label,font=font(12,True),fill=INK)
            charger=track.get("charger")
            if charger:
                x,y=p(charger["x"],charger["y"])
                r=scale*.38
                for fraction,color in [(1,"#16384d"),(.83,"#3e92b8"),(.62,"#173e58"),(.42,"#9ebfc7"),(.24,"#f2f4e9")]:
                    ry=r*(.63 if view!="topdown" else 1)*fraction
                    draw.ellipse((x-r*fraction,y-ry,x+r*fraction,y+ry),fill=color)
                draw.text((x-r*.68,y-r*.15),"CHARGE",font=font(max(7,int(scale*.075)),True),fill="#16384d")
            for station in track.get("packaging",[]):
                x,y=p(station["x"],station["y"])
                shape=station.get("shape","circle")
                r=max(6,scale*.075)
                if shape in ("triangle","Triangle","三角形"): draw.polygon([(x,y-r),(x-r,y+r),(x+r,y+r)],fill="#5a8ca5",outline="#2c5770")
                elif shape in ("square","rectangle","Square","方形"): draw.rectangle((x-r,y-r,x+r,y+r),fill="#89adc0",outline="#2c5770")
                else: draw.ellipse((x-r,y-r,x+r,y+r),fill="#afbfce",outline="#2c5770")
            self._static_cache[cache_key]=image.copy()
        else:
            image=self._static_cache[cache_key].copy()
            draw=ImageDraw.Draw(image)
        for item in world.get("obstacles",[]):
            if item.get("id") not in f.get("active_obstacles",[]):continue
            x,y=item.get("x",0),item.get("y",0)
            cube(x+item.get("w",.2)/2,y+item.get("h",.2)/2,item.get("w",.2),item.get("h",.2),.15,"#c9d6d9")
        if trails:
            for key,color in [("truth",TEAL),("odom",BLUE)]:
                pts=[p(*row[key][:2],.018) for row in frames[:index+1]]
                if len(pts)>1: draw.line(pts,fill=color,width=2)
        x,y,theta=f["truth"]
        def body_point(lateral,forward,z=0):
            return p(x+forward*math.cos(theta)-lateral*math.sin(theta),y+forward*math.sin(theta)+lateral*math.cos(theta),z)
        if sensors:
            for ray in f.get("rays",[]):
                angle=f["odom"][2]+ray["angle"]
                end=(f["odom"][0]+math.cos(angle)*ray["range"],f["odom"][1]+math.sin(angle)*ray["range"])
                draw.line([p(*f["odom"][:2],.08),p(*end,.08)],fill="#b9a36d",width=1)
            if not f.get("rays") and isinstance(f.get("front_range"),(int,float)):
                measured=f["front_range"]
                end=(x+math.cos(theta)*measured,y+math.sin(theta)*measured)
                draw.line([p(x,y,.08),p(*end,.08)],fill="#c5ae76",width=1)
            if f.get("worker_qr"):
                wx,wy=f["worker_qr"][:2]; sx,sy=p(wx,wy,.22)
                draw.rectangle((sx-8,sy-8,sx+8,sy+8),fill="#fff",outline=TEAL,width=2)
                draw.line([body_point(0,0,.08),(sx,sy)],fill=TEAL,width=1)
        # BrickPi / EV3 inspired differential-drive cart, not a full-size forklift.
        shadow=[body_point(a,b,0) for a,b in ((-.16,-.19),(.16,-.19),(.16,.20),(-.16,.20))]
        draw.polygon([(a+3,b+3) for a,b in shadow],fill="#b7c4c0")
        for side in(-1,1):
            for forward in(-.11,.11):
                px,py=body_point(side*.17,forward,.06)
                rr=max(3,scale*.06)
                draw.ellipse((px-rr,py-rr*.8,px+rr,py+rr*.8),fill="#33414a",outline="#152b36",width=1)
                draw.ellipse((px-rr*.4,py-rr*.35,px+rr*.4,py+rr*.35),fill="#e2c250")
        for z,col,half_w,back,front in[(.10,"#c6cdd0",.15,-.17,.19),(.18,"#dde4e6",.125,-.14,.14),(.21,"#38809d",.075,-.07,.07)]:
            poly=[body_point(a,b,z) for a,b in ((-half_w,back),(half_w,back),(half_w,front),(-half_w,front))]
            draw.polygon(poly,fill=col,outline="#315b6d",width=1)
        for lateral in(-.105,.105):
            draw.line([body_point(lateral,.14,.06),body_point(lateral,.34,.06)],fill="#a8ac9b",width=max(2,int(scale*.026)))
        if f.get("loaded"):
            poly=[body_point(a,b,.15) for a,b in ((-.10,.18),(.10,.18),(.10,.36),(-.10,.36))]
            draw.polygon(poly,fill="#c7ad78",outline="#856e48",width=1)
        draw.ellipse((p(x,y,.31)[0]-2,p(x,y,.31)[1]-2,p(x,y,.31)[0]+2,p(x,y,.31)[1]+2),fill="#5de0c5")
        draw.text((18,height-24),"Original course layout, redrawn | dimensions are reconstruction assumptions",font=font(11),fill=MUTED)
        return image

def render_presentation(run,index=0,size=(1360,900),view="3d"):
    image=Image.new("RGB",size,PAPER);draw=ImageDraw.Draw(image)
    w,h=size
    draw.text((35,24),"MOBILE ROBOTIK  /  ORIGINAL CHALLENGE",font=font(13,True),fill=TEAL)
    draw.text((34,51),"Mobile Robotics | Original Challenge",font=font(27,True),fill=INK)
    draw.text((35,94),"Course hardware practice 2025.09  |  Simulation reconstruction 2026.10  |  Native Python",font=font(14),fill=MUTED)
    side=330; scene_size=(w-side-55,h-177)
    scene=SceneRenderer(run).render(index,scene_size,view)
    image.paste(scene,(18,132))
    draw.rounded_rectangle((w-side-20,140,w-24,h-43),radius=16,fill="#ffffff",outline="#dce4e0")
    left=w-side+2;f=run["frames"][index]
    draw.text((left,159),"CURRENT TASK",font=font(13),fill=MUTED)
    draw.text((left,187),f"{int(f.get('task',1)):02d} / 08",font=font(29,True),fill=TEAL)
    status=english_state(f)
    words=status.split();lines=[""]
    for word in words:
        if len(lines[-1]+" "+word)>37:lines.append(word)
        else:lines[-1]=(lines[-1]+" "+word).strip()
    for line_no,line in enumerate(lines[:2]):
        draw.text((left,232+line_no*17),line,font=font(12,True),fill=INK)
    draw.line((left,268,w-48,268),fill="#e3e9e5",width=1)
    labels=run.get("tasks",[])
    active=next((i for i,t in enumerate(labels) if str(t['id'])==str(f.get('task'))),0)
    row_height=min(47,max(29,(h-470)/8))
    for i,task in enumerate(labels):
        yy=286+i*row_height
        color=TEAL if i==active else INK if i<active else MUTED
        draw.text((left,yy),f"{i+1:02d}",font=font(13,True),fill=color)
        text=TASKS_EN[i]
        draw.text((left+32,yy),text,font=font(13,i==active),fill=color)
    yy=h-164
    draw.line((left,yy,w-48,yy),fill="#e3e9e5",width=1)
    draw.text((left,yy+17),f"Simulation time   {f['t']:.1f} s",font=font(14),fill=INK)
    draw.text((left,yy+47),f"Distance            {f.get('metrics',{}).get('distance',0):.2f} m",font=font(14),fill=INK)
    draw.text((35,h-30),"2026.10 reconstruction | Differential drive and odometry | Synthetic sensor / task inputs",font=font(11),fill=MUTED)
    return image

class DesktopApp:
    """Persistent image surface, cached board, and interpolated 30 Hz replay."""
    def __init__(self,run):
        import tkinter as tk
        from tkinter import ttk
        self.tk=tk;self.run=run;self.index=0;self.playing=False
        self.clock=run["frames"][0]["t"];self.last=time.perf_counter();self.scrubbing=False
        self.photo=None;self.photo_size=None;self.last_state=None;self.last_task=None;self.last_event=None
        self.root=tk.Tk();self.root.title("Mobile Robotics | Original Challenge | Analysis")
        self.root.geometry("1400x950");self.root.minsize(1100,780);self.root.configure(bg=PAPER)
        self.renderer=SceneRenderer(run);self.view=tk.StringVar(value="Isometric overview");self.speed=tk.StringVar(value="2x")
        style=ttk.Style();style.theme_use("clam")
        style.configure("TButton",font=("Segoe UI",11),padding=8)
        style.configure("TCombobox",font=("Segoe UI",11),padding=6)
        head=tk.Frame(self.root,bg=PAPER);head.pack(fill="x",padx=25,pady=(16,10))
        tk.Label(head,text="MOBILE ROBOTIK  /  ORIGINAL CHALLENGE",font=("Segoe UI",10,"bold"),fg=TEAL,bg=PAPER).pack(anchor="w")
        tk.Label(head,text="Mobile Robotics | Original Challenge",font=("Segoe UI",23,"bold"),fg=INK,bg=PAPER).pack(anchor="w",pady=(4,2))
        tk.Label(head,text="Course hardware practice 2025.09  |  Simulation reconstruction 2026.10  |  Differential-drive learning robot",font=("Segoe UI",10),fg=MUTED,bg=PAPER).pack(anchor="w")
        toolbar=tk.Frame(self.root,bg="#17374a",padx=13,pady=10);toolbar.pack(fill="x",padx=25,pady=(0,12))
        self.play_button=ttk.Button(toolbar,text="Play",command=self.toggle);self.play_button.pack(side="left",padx=(0,7))
        ttk.Button(toolbar,text="Reset",command=self.reset).pack(side="left",padx=(0,7))
        ttk.Button(toolbar,text="Re-simulate",command=self.regenerate).pack(side="left",padx=(0,7))
        ttk.Button(toolbar,text="3D Studio",command=self.open_studio).pack(side="left",padx=(0,20))
        tk.Label(toolbar,text="VIEW",font=("Segoe UI",9),bg="#17374a",fg="white").pack(side="left",padx=(0,6))
        combo=ttk.Combobox(toolbar,textvariable=self.view,values=["Isometric overview","Top-down map"],state="readonly",width=18)
        combo.pack(side="left");combo.bind("<<ComboboxSelected>>",lambda e:self.draw(force=True))
        tk.Label(toolbar,text="SPEED",font=("Segoe UI",9),bg="#17374a",fg="white").pack(side="left",padx=(15,6))
        ttk.Combobox(toolbar,textvariable=self.speed,values=["0.5x","1x","2x","4x","8x"],state="readonly",width=5).pack(side="left")
        self.time_label=tk.Label(toolbar,text="0.0 s",font=("Consolas",11,"bold"),bg="#17374a",fg="#8dd5d7")
        self.time_label.pack(side="right",padx=8)
        body=tk.Frame(self.root,bg=PAPER);body.pack(fill="both",expand=True,padx=25)
        self.canvas=tk.Canvas(body,bg=PAPER,highlightthickness=0)
        self.canvas.pack(side="left",fill="both",expand=True,padx=(0,17))
        self.canvas_item=self.canvas.create_image(0,0,anchor="nw")
        self.resize_job=None
        self.canvas.bind("<Configure>",self.on_resize)
        side=tk.Frame(body,bg="#fff",width=320,padx=18,pady=17,highlightbackground="#dce4e0",highlightthickness=1)
        side.pack(side="right",fill="y");side.pack_propagate(False)
        tk.Label(side,text="CURRENT STATE",font=("Segoe UI",10),fg=MUTED,bg="white").pack(anchor="w")
        self.state_label=tk.Label(side,text="",font=("Segoe UI",12,"bold"),fg=TEAL,bg="white",wraplength=278,justify="left",height=3,anchor="nw")
        self.state_label.pack(fill="x",pady=(6,8))
        self.task_labels=[]
        for i,task in enumerate(run.get("tasks",[])):
            label=tk.Label(side,text=f"{task['id']:02d}   {task.get('label_en',TASKS_EN[i])}",font=("Segoe UI",11),fg=MUTED,bg="white",wraplength=274,justify="left",anchor="w",padx=8,pady=9)
            label.pack(fill="x",pady=2);self.task_labels.append(label)
        self.metrics_label=tk.Label(side,text="",font=("Segoe UI",10),bg="white",fg=INK,justify="left",anchor="w")
        self.metrics_label.pack(fill="x",pady=(17,5))
        self.event_label=tk.Label(side,text="",font=("Segoe UI",10),fg=MUTED,bg="white",wraplength=274,justify="left",anchor="nw",height=3)
        self.event_label.pack(fill="x",pady=8)
        self.seek=self.tk.Scale(self.root,from_=0,to=len(run["frames"])-1,orient="horizontal",showvalue=False,resolution=1,command=self.on_seek,bg=PAPER,highlightthickness=0,troughcolor="#d8e4e2",activebackground=TEAL)
        self.seek.pack(fill="x",padx=30,pady=(5,1))
        tk.Label(self.root,text="2026.10 reconstruction | Differential drive and odometry | Synthetic inputs | Original layout redrawn with assumed dimensions",font=("Segoe UI",9),fg=MUTED,bg=PAPER).pack(anchor="w",padx=30,pady=(2,13))
        self.root.bind("<space>",self.on_space);self.root.bind("<Home>",lambda e:self.reset())
        self.root.after(50,self.tick)

    def on_space(self,event):
        # Space already activates a focused ttk button through its class binding.
        if self.root.focus_get() and self.root.focus_get().winfo_class()=="TButton":return
        self.toggle();return "break"

    def on_resize(self,event):
        if self.resize_job:self.root.after_cancel(self.resize_job)
        self.resize_job=self.root.after(40,self.finish_resize)

    def finish_resize(self):
        self.resize_job=None;self.draw(force=True)

    def toggle(self):
        if self.index==len(self.run["frames"])-1:self.index=0;self.clock=self.run["frames"][0]["t"]
        self.playing=not self.playing;self.last=time.perf_counter()
        self.play_button.config(text="Pause" if self.playing else "Play")

    def reset(self):
        self.playing=False;self.index=0;self.clock=self.run["frames"][0]["t"]
        self.play_button.config(text="Play");self.scrubbing=True;self.seek.set(0);self.scrubbing=False
        self.last_state=None;self.last_task=None;self.draw(force=True)

    def regenerate(self):
        self.playing=False
        storage=self.run.get("warehouse_plan",{}).get("storage","i")
        self.run=load_run(regenerate=True,storage=storage);self.renderer=SceneRenderer(self.run)
        self.seek.config(to=len(self.run["frames"])-1);self.reset()

    def open_studio(self):
        subprocess.Popen([sys.executable,str(ROOT/"main.py"),"--studio"],cwd=str(ROOT))

    def on_seek(self,value):
        if self.scrubbing:return
        new_index=int(float(value))
        if new_index==self.index:return
        self.playing=False;self.play_button.config(text="Play");self.index=new_index
        self.clock=self.run["frames"][self.index]["t"];self.draw(force=True)

    def draw(self,force=False):
        if not hasattr(self,"canvas"):return
        w,h=self.canvas.winfo_width(),self.canvas.winfo_height()
        if w<50 or h<50:return
        self.index,frame=interpolated_frame(self.run,self.clock)
        view="topdown" if self.view.get().startswith("Top") else "3d"
        image=self.renderer.render(self.index,(w,h),view,sample_frame=frame)
        if self.photo is None or self.photo_size!=(w,h):
            new_photo=ImageTk.PhotoImage(image)
            self.canvas.itemconfigure(self.canvas_item,image=new_photo)
            self.photo=new_photo;self.photo_size=(w,h)
        else:
            self.photo.paste(image)
        # A single persistent canvas image avoids the delete/create blank frame.
        self.time_label.config(text=f"{frame['t']:.1f} / {self.run['frames'][-1]['t']:.1f} s")
        state=english_state(frame)
        if state!=self.last_state:
            self.state_label.config(text=state);self.last_state=state
        if frame["task"]!=self.last_task:
            for i,label in enumerate(self.task_labels):
                active=str(self.run["tasks"][i]["id"])==str(frame["task"])
                done=int(self.run["tasks"][i]["id"])<int(frame["task"])
                label.config(bg="#e8f3f0" if active else "#fff",fg=TEAL if active else INK if done else MUTED)
            self.last_task=frame["task"]
        self.metrics_label.config(text=f"Distance       {frame.get('metrics',{}).get('distance',0):.2f} m\nFront range  {frame.get('front_range',0):.2f} m\nPayload        {'Loaded' if frame.get('loaded') else 'Empty'}")
        events=[e for e in self.run.get("events",[]) if e["t"]<=frame["t"]]
        event=events[-1].get("message_en","Simulation task event") if events else "Play or scrub the timeline to inspect the run."
        if event!=self.last_event:self.event_label.config(text=event);self.last_event=event

    def tick(self):
        now=time.perf_counter();delta=now-self.last;self.last=now
        if self.playing:
            self.clock=min(self.run["frames"][-1]["t"],self.clock+delta*float(self.speed.get().replace("x","")))
            self.index,_=interpolated_frame(self.run,self.clock)
            self.scrubbing=True;self.seek.set(self.index);self.scrubbing=False;self.draw()
            if self.index==len(self.run["frames"])-1:self.playing=False;self.play_button.config(text="Play")
        self.root.after(33,self.tick)

    def mainloop(self):self.root.mainloop()
