#!/usr/bin/env python3
"""Prepare/apply a generated Doom shot without rebuilding existing scene edits.

Default is a read-only plan. Run --apply only after the new generated sheet
has been reviewed and accepted. This script submits no generation/API jobs.
"""
from pathlib import Path
from bisect import bisect_left
import argparse
import json
import subprocess

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'work'
START, END = 272, 349


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration:stream=codec_type,width,height,r_frame_rate,nb_frames',
        '-of', 'json', str(path),
    ], text=True))


def video_stream(metadata):
    return next(s for s in metadata['streams'] if s['codec_type'] == 'video')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sheet', type=Path)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--trim-border', type=int, default=2)
    args = parser.parse_args()
    poses = json.loads((WORK / 'poseguides/doom-walk.json').read_text())
    keys = [p['source_frame'] for p in poses['poses']]
    base = WORK / 'continuacao_corrigida_30fps.mp4'
    plan = {
        'read_only': not args.apply,
        'input_native': str(base),
        'sheet': str(args.sheet) if args.sheet else None,
        'source_range': [START, END],
        'source_frames': END-START,
        'final_60fps_range': [START*2, END*2],
        'anchor_frames': keys,
        'preserve_source_frames_before': START,
        'preserve_source_frames_after': 778-END,
        'total_native_frames': 778,
        'total_final_frames': 1556,
        'remaining_pending_original_range_after_success': [749, 778],
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.apply:
        return
    if args.sheet is None or not args.sheet.is_file():
        parser.error('--apply requires an accepted --sheet image')
    meta = probe(base)
    if video_stream(meta).get('nb_frames') != '778':
        raise RuntimeError('Expected latest 778-frame corrected native base')
    import cv2
    from PIL import Image
    from render_rife_backend import Rife
    from composite_consistent import OriginalBackgroundCompositor
    cv2.setNumThreads(2)
    folder = WORK / 'doom_new_ref'
    (folder / 'anchors').mkdir(parents=True, exist_ok=True)
    (folder / 'qa').mkdir(exist_ok=True)
    sheet = Image.open(args.sheet).convert('RGB')
    width, height = sheet.size
    images = {}
    for j, key in enumerate(keys):
        x, y = j % 3, j // 3
        panel = sheet.crop((round(x*width/3), round(y*height/3),
                            round((x+1)*width/3), round((y+1)*height/3)))
        border = max(0, args.trim_border)
        if border:
            panel = panel.crop((border, border, panel.width-border, panel.height-border))
        panel = panel.resize((576, 364), Image.Resampling.LANCZOS)
        path = folder / 'anchors' / f'{key:04d}.png'
        panel.save(path)
        images[key] = cv2.imread(str(path))
    patch = folder / 'doom_patch_30fps.mp4'
    encoder = subprocess.Popen([
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'bgr24',
        '-video_size', '576x854', '-framerate', '30', '-i', 'pipe:0', '-an',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-threads', '2',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(patch),
    ], stdin=subprocess.PIPE)
    rife = Rife(threads=2)
    compositor = OriginalBackgroundCompositor()
    records = []
    try:
        for frame in range(START, END):
            if frame in images:
                generated = images[frame]
            else:
                position = bisect_left(keys, frame)
                a, b = keys[position-1:position+1]
                generated = rife.between(images[a], images[b], (frame-a)/(b-a), (a, b))
            source = cv2.imread(str(WORK / 'source' / f'{frame:04d}.png'))
            if source is None:
                raise RuntimeError(f'Missing source frame {frame}')
            fixed, record = compositor.process(source[230:594], generated, frame, 'Doom')
            full = source.copy()
            full[230:594] = fixed
            if frame in keys or frame in [277, 307, 337]:
                cv2.imwrite(str(folder / 'qa' / f'{frame:04d}.jpg'), full)
            encoder.stdin.write(full.tobytes())
            records.append({'frame': frame, **record})
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('Doom patch encoder failed')
    except BaseException:
        if encoder.poll() is None:
            encoder.kill(); encoder.wait()
        raise
    if video_stream(probe(patch)).get('nb_frames') != '77':
        raise RuntimeError('Doom patch does not contain exactly 77 frames')
    patched_native = WORK / 'continuacao_com_doom_30fps.mp4'
    graph = (
        '[0:v]trim=start_frame=0:end_frame=272,setpts=PTS-STARTPTS[a];'
        '[1:v]trim=start_frame=0:end_frame=77,setpts=PTS-STARTPTS[b];'
        '[0:v]trim=start_frame=349:end_frame=778,setpts=PTS-STARTPTS[c];'
        '[a][b][c]concat=n=3:v=1:a=0,fps=30,setsar=1[v]'
    )
    subprocess.run([
        'ffmpeg', '-y', '-v', 'error', '-i', str(base), '-i', str(patch),
        '-filter_complex', graph, '-map', '[v]', '-frames:v', '778', '-an',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '16', '-threads', '2',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(patched_native),
    ], check=True)
    if video_stream(probe(patched_native)).get('nb_frames') != '778':
        raise RuntimeError('Doom insertion changed full timeline duration')
    final = Path('/workspace/shared/downloads/video-com-doom-em-revisao-60fps.mp4')
    subprocess.run(['python', str(ROOT / 'export_adjacent60.py'),
                    str(patched_native), str(final)], check=True)
    final_meta = probe(final)
    stream = video_stream(final_meta)
    if stream.get('nb_frames') != '1556' or stream.get('r_frame_rate') != '60/1':
        raise RuntimeError('Wrong full final frame count or FPS')
    if any(s['codec_type'] == 'audio' for s in final_meta['streams']):
        raise RuntimeError('Expected silent final')
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(final), '-f', 'null', '-'], check=True)
    (folder / 'insertion_report.json').write_text(json.dumps({
        'plan': plan, 'metadata': final_meta, 'qa': records,
        'archive_update_still_needed': True,
        'remaining_pending_original_ranges': [[749, 778]],
    }, indent=2))
    print(json.dumps({'final': str(final), 'metadata': final_meta,
        'archive_status': 'Update Doom-pending README and status before refreshing ZIP'}, indent=2))


if __name__ == '__main__':
    main()
