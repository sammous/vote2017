#Source: Gensim
from __future__ import division
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import unicodedata
import logging
import string
import re

from timeit import default_timer

logger = logging.getLogger(__name__)

RE_PUNCT = re.compile('([%s])+' % re.escape(string.punctuation), re.UNICODE)
def strip_punctuation(s):
    s = to_unicode(s)
    return RE_PUNCT.sub(" ", s)

RE_TAGS = re.compile(r"<([^>]+)>", re.UNICODE)
def strip_tags(s):
    s = to_unicode(s)
    return RE_TAGS.sub("",s)

def strip_short(s, minsize=3):
    s = to_unicode(s)
    return " ".join(e for e in s.split() if len(e) >= minsize)

RE_NUMERIC = re.compile(r"[0-9]+", re.UNICODE)
def strip_numeric(s):
    s = to_unicode(s)
    return RE_NUMERIC.sub("_numeric_", s)

RE_NONALPHA = re.compile(r"\W", re.UNICODE)
def strip_non_alphanum(s):
    s = to_unicode(s)
    return RE_NONALPHA.sub(" ", s)

RE_WHITESPACE = re.compile(r"(\s)+", re.UNICODE)
def strip_multiple_whitespaces(s):
    s = to_unicode(s)
    return RE_WHITESPACE.sub(" ", s)

RE_AL_NUM = re.compile(r"([a-z]+)([0-9]+)", flags=re.UNICODE)
RE_NUM_AL = re.compile(r"([0-9]+)([a-z]+)", flags=re.UNICODE)
def split_alphanum(s):
    s = to_unicode(s)
    s = RE_AL_NUM.sub(r"\1 \2", s)
    return RE_NUM_AL.sub(r"\1 \2", s)

def strip_accent(s):
    nkfd_form = unicodedata.normalize('NFKD', s)
    return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])

DEFAULT_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces,
                   strip_numeric, strip_short]

def read_file(path):
    with smart_open(path) as fin:
        for line in fin.readlines():
            yield line

def write_file(path, sentences):
    with smart_open(path, 'wb') as f:
        print('writing file on path %s' %path)
        for sentence in sentences:
            sentence = map(lambda w: w.encode('utf8'), sentence)
            sentence = [w for w in sentence if w.strip()]
            f.write(','.join(sentence) + '\n')
    print('file written successfully.')
    f.close()

def to_unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode."""
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)
    
def preprocess_sentence(sentence, filters=DEFAULT_FILTERS):
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    sentence = to_unicode(sentence)
    for f in filters:
        if sentence:
            sentence = f(sentence)
    return tokenizer.tokenize(sentence)

def preprocess_doc(seq):
    print('preprocessing...')
    preprocessed_lines = []
    total_progress = 0
    size = len(list(seq))
    for ix, line in enumerate(seq):
        print(ix)
        progress = int((float(ix)/size)*100)
        line = preprocess_sentence(line)
        if total_progress != progress:
            total_progress = progress
            print('%.0f%% processed...' %progress)
        preprocessed_lines.append(line)
    return preprocessed_lines

if __name__ == '__main__':
    pass
