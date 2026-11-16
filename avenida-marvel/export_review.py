#!/usr/bin/env python3
"""Assemble the completed review windows on the source video's original canvas."""
from pathlib import Path
import json
import subprocess

import cv2

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'work'
OUTPUT = Path('/workspace/shared/downloads/previa-marvel-sylvie-olhar-ajustado-60fps.mp4')


def main():
    plan = json.loads((WORK / 'review_indices.json').read_text())
    indices = [i for start, end in plan['ranges'] for i in range(2 * start, 2 * end)]
    missing = [i for i in indices if not (WORK / 'review_rife' / f'{i:04d}.png').exists()]
    if missing:
        raise RuntimeError(f'Missing {len(missing)} rendered frames: {missing[:10]}')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(OUTPUT.stem + '.encoding.mp4')
    command = [
        'ffmpeg', '-y', '-v', 'warning',
        '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', '576x854',
        '-framerate', '60', '-i', 'pipe:0',
        '-vf', 'scale=720:1068:flags=lanczos,setsar=1',
        '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        '-metadata', 'title=Prévia parcial com Sylvie e olhar ajustado',
        '-metadata', 'comment=Trechos recriados e interpolados a 60 fps, sem áudio.',
        str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    previous_source = None
    original = None
    try:
        for number, index in enumerate(indices):
            source_index = index // 2
            if previous_source != source_index:
                original = cv2.imread(str(WORK / 'source' / f'{source_index:04d}.png'))
                if original is None or original.shape != (854, 576, 3):
                    raise RuntimeError(f'Invalid source frame {source_index}')
                previous_source = source_index
            active = cv2.imread(str(WORK / 'review_rife' / f'{index:04d}.png'))
            if active is None or active.shape != (364, 576, 3):
                raise RuntimeError(f'Invalid edited frame {index}')
            full = original.copy()
            full[230:594, :, :] = active
            process.stdin.write(full.tobytes())
            if number % 120 == 0:
                print(f'Encoded {number}/{len(indices)} frames', flush=True)
        process.stdin.close()
        returncode = process.wait()
        if returncode:
            raise RuntimeError(f'ffmpeg exited {returncode}')
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise
    temporary.replace(OUTPUT)
    print(f'Completed {len(indices)} frames, {len(indices)/60:.2f}s: {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
