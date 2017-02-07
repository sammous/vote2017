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

STOP_WORDS = set("""
a à â abord absolument afin ah ai aie ailleurs ainsi ait allaient allo allons
allô alors anterieur anterieure anterieures apres après as assez attendu au
aucun aucune aujourd aujourd'hui aupres auquel aura auraient aurait auront
aussi autre autrefois autrement autres autrui aux auxquelles auxquels avaient
avais avait avant avec avoir avons ayant
bah bas basee bat beau beaucoup bien bigre boum bravo brrr
ça car ce ceci cela celle celle-ci celle-là celles celles-ci celles-là celui
celui-ci celui-là cent cependant certain certaine certaines certains certes ces
cet cette ceux ceux-ci ceux-là chacun chacune chaque cher chers chez chiche
chut chère chères ci cinq cinquantaine cinquante cinquantième cinquième clac
clic combien comme comment comparable comparables compris concernant contre
couic crac
da dans de debout dedans dehors deja delà depuis dernier derniere derriere
derrière des desormais desquelles desquels dessous dessus deux deuxième
deuxièmement devant devers devra different differentes differents différent
différente différentes différents dire directe directement dit dite dits divers
diverse diverses dix dix-huit dix-neuf dix-sept dixième doit doivent donc dont
douze douzième dring du duquel durant dès désormais
effet egale egalement egales eh elle elle-même elles elles-mêmes en encore
enfin entre envers environ es ès est et etaient étaient etais étais etait était
etant étant etc été etre être eu euh eux eux-mêmes exactement excepté extenso
exterieur
fais faisaient faisant fait façon feront fi flac floc font
gens
ha hein hem hep hi ho holà hop hormis hors hou houp hue hui huit huitième hum
hurrah hé hélas i il ils importe
je jusqu jusque juste
la laisser laquelle las le lequel les lesquelles lesquels leur leurs longtemps
lors lorsque lui lui-meme lui-même là lès
ma maint maintenant mais malgre malgré maximale me meme memes merci mes mien
mienne miennes miens mille mince minimale moi moi-meme moi-même moindres moins
mon moyennant multiple multiples même mêmes
na naturel naturelle naturelles ne neanmoins necessaire necessairement neuf
neuvième ni nombreuses nombreux non nos notamment notre nous nous-mêmes nouveau
nul néanmoins nôtre nôtres
o ô oh ohé ollé olé on ont onze onzième ore ou ouf ouias oust ouste outre
ouvert ouverte ouverts où
paf pan par parce parfois parle parlent parler parmi parseme partant
particulier particulière particulièrement pas passé pendant pense permet
personne peu peut peuvent peux pff pfft pfut pif pire plein plouf plus
plusieurs plutôt possessif possessifs possible possibles pouah pour pourquoi
pourrais pourrait pouvait prealable precisement premier première premièrement
pres probable probante procedant proche près psitt pu puis puisque pur pure
qu quand quant quant-à-soi quanta quarante quatorze quatre quatre-vingt
quatrième quatrièmement que quel quelconque quelle quelles quelqu'un quelque
quelques quels qui quiconque quinze quoi quoique
rare rarement rares relative relativement remarquable rend rendre restant reste
restent restrictif retour revoici revoilà rien
sa sacrebleu sait sans sapristi sauf se sein seize selon semblable semblaient
semble semblent sent sept septième sera seraient serait seront ses seul seule
seulement si sien sienne siennes siens sinon six sixième soi soi-même soit
soixante son sont sous souvent specifique specifiques speculatif stop
strictement subtiles suffisant suffisante suffit suis suit suivant suivante
suivantes suivants suivre superpose sur surtout
ta tac tant tardive te tel telle tellement telles tels tenant tend tenir tente
tes tic tien tienne tiennes tiens toc toi toi-même ton touchant toujours tous
tout toute toutefois toutes treize trente tres trois troisième troisièmement
trop très tsoin tsouin tu té
un une unes uniformement unique uniques uns
va vais vas vers via vif vifs vingt vivat vive vives vlan voici voilà vont vos
votre vous vous-mêmes vu vé vôtre vôtres
zut
""".split())

RE_PUNCT = re.compile('([%s])+' % re.escape(string.punctuation), re.UNICODE)
def strip_punctuation(s):
    s = to_unicode(s)
    return RE_PUNCT.sub(" ", s)

RE_TAGS = re.compile(r"<([^>]+)>", re.UNICODE)
def strip_tags(s):
    s = to_unicode(s)
    return RE_TAGS.sub("",s)

def strip_short(s, minsize=5):
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

def strip_url(s):
    s = re.sub(ur'^((http|https):\/\/\S+)', ur'_link_', s)
    s = re.sub(ur'\s((http|https):\/\/\S+)', ur' _link_', s)
    return s

DEFAULT_FILTERS = [lambda x: x.lower(), strip_url, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_short]

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
