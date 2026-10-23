#!/usr/bin/env python3
"""Export media from the same newly implemented original-layout simulator data."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from native_view import load_run,render_presentation

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path)
    parser.add_argument("--output",type=Path,default=ROOT/"docs/media")
    parser.add_argument("--frames",type=int,default=90)
    parser.add_argument("--fps",type=int,default=9)
    parser.add_argument("--ffmpeg",help="Optional explicit ffmpeg path for MP4")
    args=parser.parse_args();run=load_run(args.data);args.output.mkdir(parents=True,exist_ok=True)
    count=max(2,args.frames)
    indices=[round(i*(len(run['frames'])-1)/(count-1)) for i in range(count)]
    images=[render_presentation(run,index,size=(1080,760)) for index in indices]
    # Fixed palette derived from the first scene minimizes video-wide color flicker.
    palette=images[0].quantize(colors=192)
    frames=[im.quantize(palette=palette) for im in images]
    gif=args.output/"original-challenge-reconstruction.gif"
    durations=[round(1000/args.fps)]*len(frames);durations[-1]=1800
    frames[0].save(gif,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=True)
    print(gif)
    render_presentation(run,round((len(run['frames'])-1)*.65)).save(args.output/"original-challenge-overview.png")
    render_presentation(run,round((len(run['frames'])-1)*.65),view="topdown").save(args.output/"original-challenge-topdown.png")
    ffmpeg=args.ffmpeg or shutil.which("ffmpeg")
    if ffmpeg:
        video_frames=images+[images[-1]]*round(args.fps*1.8)
        out=args.output/"original-challenge-reconstruction.mp4"
        w,h=images[0].size
        command=[str(ffmpeg),"-y","-loglevel","error","-f","rawvideo","-vcodec","rawvideo","-pix_fmt","rgb24","-s",f"{w}x{h}","-r",str(args.fps),"-i","-","-an","-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(out)]
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        for im in video_frames:process.stdin.write(im.convert("RGB").tobytes())
        process.stdin.close()
        stderr=process.stderr.read().decode(errors="replace")
        code=process.wait()
        if code:raise RuntimeError(f"ffmpeg exited {code}: {stderr}")
        print(out)
    else:print("ffmpeg not found; GIF and PNG exports complete. Supply --ffmpeg for MP4.")

if __name__=="__main__":main()
