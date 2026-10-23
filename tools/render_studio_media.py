"""Record the native OpenGL scene from the actual generated Challenge trajectory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from native_3d import ChallengeStudio, load_recording
from PIL import Image

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ffmpeg',default=shutil.which('ffmpeg'))
    parser.add_argument('--frames',type=int,default=240)
    parser.add_argument('--fps',type=int,default=16)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/media')
    args=parser.parse_args()
    if not args.ffmpeg:parser.error('Supply --ffmpeg <path> or install ffmpeg')
    args.output.mkdir(parents=True,exist_ok=True)
    run=load_recording()
    app=ChallengeStudio(run,offscreen=True,quality='full',size=(1280,800))
    app.playing=False
    for _ in range(6):app.graphicsEngine.renderFrame()
    video=args.output/'studio-preview.mp4'
    encoder=subprocess.Popen([str(args.ffmpeg),'-y','-f','rawvideo','-vcodec','rawvideo',
        '-pix_fmt','rgb24','-s','1280x800','-r',str(args.fps),'-i','-',
        '-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(video)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    gif_images=[]
    try:
        for i in range(args.frames):
            app.clock=app.duration*i/max(1,args.frames-1)
            app._update_scene(force=False)
            app.graphicsEngine.renderFrame()
            texture=app.win.getScreenshot()
            frame=Image.frombytes('RGB',(texture.getXSize(),texture.getYSize()),bytes(texture.getRamImageAs('RGB')))
            frame=frame.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            encoder.stdin.write(frame.tobytes())
            if i%3==0 or i==args.frames-1:gif_images.append(frame.resize((960,600)))
        for _ in range(round(args.fps*1.5)):encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        err=encoder.stderr.read();code=encoder.wait()
        if code:raise RuntimeError(err.decode(errors='replace')[-1500:])
        palette=gif_images[0].quantize(colors=192)
        frames=[im.quantize(palette=palette) for im in gif_images]
        delay=round(1000*3/args.fps)
        frames[0].save(args.output/'studio-preview.gif',save_all=True,append_images=frames[1:],
                       duration=[delay]*(len(frames)-1)+[1500],loop=0,optimize=True)
        app.save_frame(args.output/'studio-overview.png',progress=.57,camera_mode='Overview')
        app.save_frame(args.output/'studio-robot.png',progress=.45,camera_mode='Robot focus')
        subprocess.run([str(args.ffmpeg),'-v','error','-i',str(video),'-f','null','-'],check=True,capture_output=True)
        report={'renderer':'Panda3D 1.10.16 / native OpenGL','recording_frames':len(run['frames']),
                'simulation_duration_seconds':app.duration,'video_frames':args.frames+round(args.fps*1.5),
                'fps':args.fps,'video_duration_seconds':args.frames/args.fps+1.5,'final_hold_seconds':1.5,
                'language':'English','mp4_decode_passed':True,
                'source_sha256':hashlib.sha256((ROOT/'docs/data/original-challenge.json').read_bytes()).hexdigest()}
        (args.output/'studio-media-metadata.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report))
    finally:
        if encoder.poll() is None:encoder.kill()
        app.destroy()

if __name__=='__main__':main()
