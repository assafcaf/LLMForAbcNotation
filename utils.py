
USEABLE_KEYS = [i+":" for i in "BCDFGHIKLMmNOPQRrSsTUVWwXZ"]
def parse_notation(sample):
    abc_notation = sample["abc notation"]
    abc_notation = [l for l in abc_notation.split("\n") if not l.startswith("%")]
    for i, field in enumerate(abc_notation):
        if sum([field.startswith(f) for f in USEABLE_KEYS]):
            continue
        break
    keys = "\n".join(abc_notation[:i])
    abc = "".join(abc_notation[i:]).replace("x8", "").replace("||", "")
    return keys, abc, sample['control code']


def chunk_iterator(iterator, chunk_size):
    chunk = []
    for item in iterator:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk