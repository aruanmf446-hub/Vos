"""Bounded GGUF metadata reader, stdlib only. Does not dequantize/load weights."""
import io
import struct


def read_header(content):
    stream=io.BytesIO(content)
    def read(n):
        value=stream.read(n)
        if len(value)!=n:raise ValueError('Cabecalho GGUF incompleto')
        return value
    def number(fmt):return struct.unpack('<'+fmt,read(struct.calcsize('<'+fmt)))[0]
    def string():
        n=number('Q')
        if n>32*1024**2:raise ValueError('String GGUF excessiva')
        return read(n).decode('utf-8')
    types={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
    def value(kind):
        if kind==8:return string()
        if kind==9:
            subtype=number('I');count=number('Q')
            if count>1000000:raise ValueError('Array GGUF excessivo')
            return [value(subtype) for _ in range(count)]
        if kind not in types:raise ValueError('Tipo GGUF desconhecido')
        return number(types[kind])
    if read(4)!=b'GGUF' or number('I')!=3:raise ValueError('Esperado GGUF v3')
    count=number('Q');fields=number('Q')
    if count>100000 or fields>1000:raise ValueError('GGUF excessivo')
    metadata={}
    for _ in range(fields):name=string();metadata[name]=value(number('I'))
    tensors={}
    for _ in range(count):
        name=string();ndim=number('I')
        if ndim>5:raise ValueError('Tensor GGUF invalido')
        shape=[number('Q') for _ in range(ndim)];kind=number('I');offset=number('Q')
        tensors[name]={'shape':shape,'type':kind,'offset':offset}
    return {'metadata':metadata,'tensors':tensors,'header_bytes':stream.tell()}


def require_animate(header):
    tensors=header['tensors']
    if header['metadata'].get('general.architecture')!='wan' or 'pose_patch_embedding.weight' not in tensors or not any('face' in k for k in tensors):
        raise ValueError('GGUF nao identifica Wan Animate com pose e rosto')
    if not any(t['type']==11 for t in tensors.values()):raise ValueError('GGUF sem tensores Q3_K')
