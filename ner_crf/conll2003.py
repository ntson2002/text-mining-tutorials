from nltk.corpus.reader import ConllCorpusReader
from ner_crf.utils import train_crf, evaluate, sent2features
import pickle
import nltk


def load_data_conll2003():
    train_sents = ConllCorpusReader('../data/conll2003', 'train.txt', ['words', 'pos', 'ignore', 'chunk']).iob_sents()
    test_sents = ConllCorpusReader('../data/conll2003', 'valid.txt', ['words', 'pos', 'ignore', 'chunk']).iob_sents()

    train_sents = [x for x in train_sents if len(x) > 0]
    test_sents = [x for x in test_sents if len(x) > 0]

    print("#train_sents", len(train_sents))
    print("#test_sents", len(test_sents))
    return train_sents, test_sents


def text_to_conll(text):
    text = nltk.word_tokenize(text)
    x = nltk.pos_tag(text)
    x = [i + ('-',) for i in x]
    return x


def test():
    # Evaluate
    print("===== Load and evaluate model ====")
    saved = "./models/ner_conll2003.pickle"
    train_sents, test_sents = load_data_conll2003()
    print(test_sents[0])
    crf = pickle.load(open(saved, 'rb'))
    evaluate(crf, test_sents)
    evaluate(crf, train_sents)

    # Predict from text
    print("===== Predict from text ====")
    text = "He is a German who works at Google Inc."
    a_sentence = text_to_conll(text)
    print(a_sentence)
    a_sentence_features = sent2features(a_sentence)
    yyy = crf.predict([a_sentence_features])
    print(yyy)


def train():
    print("Train conll 2003 (NER English dataset)")
    saved = "./models/ner_conll2003.pickle"
    train_sents, test_sents = load_data_conll2003()
    crf = train_crf(train_sents, saved)
    evaluate(crf, test_sents)


if __name__ == '__main__':
    # train()
    test()


