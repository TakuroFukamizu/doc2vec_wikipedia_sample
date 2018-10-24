

from collections import namedtuple
from gensim.models.doc2vec import Doc2Vec
import wikipedia
import nltk

# -----------------------------------

def wikipage2words(word_as_id):
    wikipedia.set_lang("en")
    page = wikipedia.page(word_as_id)
    sentence = page.content
    words = nltk.word_tokenize(sentence)
    return words

def get_words(item):
    if item.type == 'wikipedia':
        return wikipage2words(item.id)
    elif item.type == 'local_file':
        with open(item.id) as f:
            return f.read()
    else:
        return None

def get_similarity(model, item1, item2):
    doc1_words = get_words(item1)
    doc2_words = get_words(item2)

    sim_value = model.docvecs.similarity_unseen_docs(
        model,
        doc1_words,
        doc2_words,
        alpha=1,
        min_alpha=0.0001,
        steps=5
        )
    return sim_value

# -----------------------------------

Item = namedtuple('Item', ('type', 'label', 'id'))

nltk.download('punkt')

if __name__ == '__main__':
    # model = Doc2Vec.load('model/doc2vec.model')
    model = Doc2Vec.load('model/enwiki_dbow/doc2vec.bin')

    items = [
        (
            Item(type='wikipedia', label='Apple_Inc', id='Apple_Inc'),
            Item(type='wikipedia', label='Google', id='Google'),
        ),
        (
            Item(type='wikipedia', label='Apple_Inc', id='Apple_Inc'),
            Item(type='wikipedia', label='Renault', id='Renault'),
        ),
        (
            Item(type='wikipedia', label='Alphabet Inc.', id='Alphabet Inc.'),
            Item(type='wikipedia', label='Alphabet', id='Alphabet'),
        ),
        (
            Item(type='wikipedia', label='Apple_Inc', id='Apple_Inc'),
            Item(type='local_file', label='CNN - iPhone XR review', id='./article_samples/cnn_20181023_tech_iphone-xr-review.txt'),
        ),
        (
            Item(type='local_file', label='Apple_Inc dict', id='./article_samples/apple_inc.txt'),
            Item(type='local_file', label='CNN - iPhone XR review', id='./article_samples/cnn_20181023_tech_iphone-xr-review.txt'),
        ),
        (
            Item(type='wikipedia', label='Google', id='Google'),
            Item(type='local_file', label='CNN - Google should buy Twitter and Square. But it won\'t', id='./article_samples/cnn_20181019_tech_alphabet-google-twitter-square-jack-dorsey.txt'),
        ),
        (
            Item(type='local_file', label='Google dict', id='./article_samples/google.txt'),
            Item(type='local_file', label='CNN - Google should buy Twitter and Square. But it won\'t', id='./article_samples/cnn_20181019_tech_alphabet-google-twitter-square-jack-dorsey.txt'),
        ),
        (
            Item(type='wikipedia', label='Twitter', id='Twitter'),
            Item(type='local_file', label='CNN - Google should buy Twitter and Square. But it won\'t', id='./article_samples/cnn_20181019_tech_alphabet-google-twitter-square-jack-dorsey.txt'),
        ),
        (
            Item(type='local_file', label='Twitter dict', id='./article_samples/twitter.txt'),
            Item(type='local_file', label='CNN - Google should buy Twitter and Square. But it won\'t', id='./article_samples/cnn_20181019_tech_alphabet-google-twitter-square-jack-dorsey.txt'),
        ),
    ]

    print("# ----------------------------------- #")

    for item1, item2 in items:
        sim_value = get_similarity(model, item1, item2)
        print("{} vs {} = {}".format(item1.label, item2.label, sim_value))
    
    print("# ----------------------------------- #")

