"""All editing decisions are exact rational intervals; filenames never order frames."""
from fractions import Fraction
from bisect import bisect_right
from math import ceil, lcm
from .errors import StudioError

SCHEMA = 'vos-frame-studio/1'


def rational(value):
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError, TypeError):
        raise StudioError('Valor de tempo ou FPS inválido.')


def scenes_for(frames, duration, cuts=()):
    cuts = list(cuts)
    if cuts != sorted(set(cuts)) or any(type(n) is not int or n <= 0 or n >= len(frames) for n in cuts):
        raise StudioError('Cortes devem ser índices únicos, em ordem, entre o segundo e o último quadro.')
    boundaries = [0, *cuts, len(frames)]
    return [{'id': f'scene_{i+1:03d}', 'order': i, 'first_frame': a, 'end_frame': b,
             'start': frames[a]['start'], 'end': frames[b]['start'] if b < len(frames) else str(duration),
             'expected_native_frames': b-a} for i, (a,b) in enumerate(zip(boundaries, boundaries[1:]))]


def scene_at(project, source_index):
    return next(s['id'] for s in project['scenes'] if s['first_frame'] <= source_index < s['end_frame'])


def native_timeline(project):
    return [{'start': f['start'], 'duration': f['duration'], 'source_index': f['index'],
             'scene_id': scene_at(project, f['index']), 'asset': {'kind': 'original', 'index': f['index']}}
            for f in project['frames']]


def select_bounds(project, selection):
    scope = selection.get('scope', 'full')
    if scope in ['full', 'all_scenes']:
        a, b = 0, len(project['frames'])
    elif scope == 'scene':
        scene = next((s for s in project['scenes'] if s['id'] == selection.get('scene_id')), None)
        if scene is None:
            raise StudioError('Cena não encontrada.')
        a, b = scene['first_frame'], scene['end_frame']
    elif scope == 'range':
        a, b = selection.get('first_frame'), selection.get('end_frame')
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(project['frames']):
            raise StudioError('Trecho inválido: o fim é exclusivo e deve ser posterior ao início.')
    else:
        raise StudioError('Selecione vídeo completo, cena ou trecho.')
    start = rational(project['frames'][a]['start'])
    end = rational(project['frames'][b]['start']) if b < len(project['frames']) else rational(project['source']['duration'])
    return a, b, start, end


def slice_timeline(timeline, begin, end):
    result = []
    for f in timeline:
        start = rational(f['start']); finish = start + rational(f['duration'])
        a, b = max(start, begin), min(finish, end)
        if a < b:
            result.append({**f, 'start': str(a), 'duration': str(b-a)})
    return result


def validate_coverage(timeline, begin, end):
    cursor = begin
    for f in timeline:
        if rational(f['start']) != cursor or rational(f['duration']) <= 0:
            raise StudioError('A linha do tempo contém uma lacuna ou sobreposição. A montagem foi bloqueada.', code='timeline_gap')
        cursor += rational(f['duration'])
    if cursor != end:
        raise StudioError('A linha do tempo não cobre todo o intervalo solicitado. A montagem foi bloqueada.', code='timeline_gap')


def replace_interval(timeline, frames, begin, end, total):
    validate_coverage(frames, begin, end)
    result = slice_timeline(timeline, Fraction(0), begin) + frames + slice_timeline(timeline, end, total)
    validate_coverage(result, Fraction(0), total)
    return result


def resample(timeline, project, begin, end, fps):
    """Temporal image selection only. Preserve each scene boundary and total duration."""
    if fps == 'native':
        return slice_timeline(timeline, begin, end)
    rate = rational(fps)
    if rate not in [Fraction(15), Fraction(24), Fraction(25), Fraction(30), Fraction(60)]:
        raise StudioError('FPS permitido: original, 15, 24, 25, 30 ou 60.')
    boundaries = sorted({begin, end, *(rational(s['start']) for s in project['scenes'] if begin < rational(s['start']) < end)})
    starts = [rational(f['start']) for f in timeline]
    result = []
    for a,b in zip(boundaries, boundaries[1:]):
        count = ceil((b-a)*rate)
        for i in range(count):
            at = a + Fraction(i, 1)/rate
            finish = min(b, a + Fraction(i+1, 1)/rate)
            index = max(0, bisect_right(starts, at)-1)
            # Preserve the last visible image of each scene, including the final
            # image of a downsampled range. Any removal is reported explicitly.
            if i == count-1:
                index = max(0, bisect_right(starts, b-Fraction(1,10**12))-1)
            result.append({**timeline[index], 'start': str(at), 'duration': str(finish-at)})
    validate_coverage(result, begin, end)
    return result


def sampling_report(original, sampled, fps):
    key = lambda f: (f['asset']['kind'], f['asset'].get('import_id', ''), f['asset'].get('path', f['source_index']))
    available = {key(f) for f in original}
    selected = [key(f) for f in sampled]
    unique = set(selected)
    return {'mode': fps, 'input_frames': len(original), 'output_frames': len(sampled),
            'discarded_frames': len(available - unique), 'duplicated_frames': len(selected)-len(unique),
            'note': 'Amostragem temporal: pode descartar ou repetir imagens; não cria movimento novo. Cortes e duração são mantidos, com intervalos fracionados quando necessário.' if fps != 'native' else 'Todos os quadros, com a cadência preservada.'}


def clock_for(frames, begin):
    clock = 1
    for f in frames:
        clock = lcm(clock, (rational(f['start'])-begin).denominator, rational(f['duration']).denominator)
        if clock > 1_000_000_000:
            raise StudioError('A base de tempo exige precisão além do MP4 suportado. Use a cadência original ou outro arquivo.', code='unsupported_time_base')
    return clock
