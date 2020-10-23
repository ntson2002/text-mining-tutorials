import nltk
import pickle
from ner_crf.utils import train_crf, evaluate


def load_data():
    nltk.download('conll2002')

    print("fileids", nltk.corpus.conll2002.fileids())
    train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
    test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))

    train_sents = [x for x in train_sents if len(x) > 0]
    test_sents = [x for x in test_sents if len(x) > 0]

    print("#train_sents", len(train_sents))
    print("#test_sents", len(test_sents))
    return train_sents, test_sents


def train():
    print("Train conll 2002 (NER Spain dataset)")
    train_sents, test_sents = load_data()
    saved = "./models/ner_conll2002.pickle"
    crf = train_crf(train_sents, saved)
    evaluate(crf, test_sents)


def test():
    # Evaluate
    saved = "./models/ner_conll2002.pickle"
    train_sents, test_sents = load_data()
    print(test_sents[0])
    crf = pickle.load(open(saved, 'rb'))
    evaluate(crf, test_sents)
    evaluate(crf, train_sents)


if __name__ == '__main__':
    train()
    # test()


