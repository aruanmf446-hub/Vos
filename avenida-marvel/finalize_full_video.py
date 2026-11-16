#!/usr/bin/env python3
"""Splice approved local fixes, export full 60 FPS video and refresh archive."""
from pathlib import Path
import json, subprocess

ROOT=Path(__file__).resolve().parent
WORK=ROOT/'work'


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type,width,height,r_frame_rate,nb_frames','-of','json',str(path)],text=True))


def main():
    native=WORK/'continuacao_30fps.mp4'
    stand=WORK/'stand_fix_30fps.mp4'
    flash=WORK/'flash_fix_30fps.mp4'
    for path,count in [(native,778),(stand,37),(flash,2)]:
        meta=probe(path)
        video=next(s for s in meta['streams'] if s['codec_type']=='video')
        if int(video['nb_frames'])!=count or video['r_frame_rate']!='30/1':
            raise RuntimeError(f'Wrong patch length/FPS: {path}')
    corrected=WORK/'continuacao_corrigida_30fps.mp4'
    graph=(
        '[0:v]trim=start_frame=0:end_frame=39,setpts=PTS-STARTPTS[a];'
        '[1:v]trim=start_frame=0:end_frame=37,setpts=PTS-STARTPTS[b];'
        '[0:v]trim=start_frame=76:end_frame=747,setpts=PTS-STARTPTS[c];'
        '[2:v]trim=start_frame=0:end_frame=2,setpts=PTS-STARTPTS[d];'
        '[0:v]trim=start_frame=749:end_frame=778,setpts=PTS-STARTPTS[e];'
        '[a][b][c][d][e]concat=n=5:v=1:a=0,fps=30,setsar=1[v]'
    )
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(native),'-i',str(stand),'-i',str(flash),
        '-filter_complex',graph,'-map','[v]','-frames:v','778','-an','-c:v','libx264',
        '-preset','fast','-crf','16','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(corrected)],check=True)
    if next(s for s in probe(corrected)['streams'] if s['codec_type']=='video')['nb_frames']!='778':
        raise RuntimeError('Splice changed timeline length')
    final=Path('/workspace/shared/downloads/video-continuo-em-revisao-60fps.mp4')
    subprocess.run(['python',str(ROOT/'export_adjacent60.py'),str(corrected),str(final)],check=True)
    meta=probe(final)
    videos=[s for s in meta['streams'] if s['codec_type']=='video']
    if len(videos)!=1 or videos[0]['nb_frames']!='1556' or videos[0]['r_frame_rate']!='60/1':
        raise RuntimeError('Wrong final frame count or FPS')
    if any(s['codec_type']=='audio' for s in meta['streams']):raise RuntimeError('Expected silent video')
    subprocess.run(['ffmpeg','-v','error','-i',str(final),'-f','null','-'],check=True)
    report=json.loads((WORK/'full_timeline_status.json').read_text())
    report['final_adjustments']={
        'stand_filtered_keys':[57,59,61,63],
        'stand_fix_source_range':[39,76],
        'flash_hold_source_range':[747,749],
        'flash_hold_anchor':746,
        'pending_original_ranges':[[272,349],[749,778]],
        'final_frames':1556,
        'final_fps':60,
        'final_metadata':meta,
    }
    (WORK/'full_timeline_status.json').write_text(json.dumps(report,indent=2))
    subprocess.run(['python',str(ROOT/'update_chat_archive.py')],check=True)
    print(json.dumps({'completed':str(final),'metadata':meta,'pending_original_ranges':report['final_adjustments']['pending_original_ranges']},indent=2),flush=True)


if __name__=='__main__':main()
