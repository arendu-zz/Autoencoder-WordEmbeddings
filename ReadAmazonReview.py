__author__ = 'arenduchintala'
import gzip
import simplejson
import pdb
from collections import defaultdict
def parse(filename):
    f = gzip.open(filename, 'r')
    entry = {}
    for l in f:
        l = l.strip()
        colonPos = l.find(':')
        if colonPos == -1:
            yield entry
            entry = {}
            continue
        attribute_name  = l[:colonPos]
        attribute_val = l[colonPos+2:]
        entry[attribute_name] = attribute_val
    yield entry

if __name__ == '__main__':
    # script here
    funcwords = set(open('functionwords.txt','r').read().split('\n'))
    cooccurance = defaultdict(set)
    vocab_id = {}
    w = 5
    for e in parse('Arts.txt.gz'):
        #print simplejson.dumps(e)
        if 'review/text' in e: 
            txt = e['review/text'].lower().split()
            for idx, token in enumerate(txt):
                start = idx-w if idx-w > 0 else 0
                end = idx+w if idx+w < len(txt) else len(txt)
                window = set(txt[start:end]) - funcwords
                if token not in vocab_id:
                    vocab_id[token] = len(vocab_id)
                cooccurance[token].update(window)
    for t in cooccurance.keys():
        vec_cooccurance = [str(vocab_id[s]) for s in cooccurance[t]]
        print t, vocab_id[t], ','.join(vec_cooccurance)
