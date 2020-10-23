import nltk
import pickle
import sklearn_crfsuite
from nltk.corpus.reader import ConllCorpusReader
from sklearn_crfsuite import metrics


def load_data_vslp2018():
    train_sents = ConllCorpusReader('../data/vlsp2018', 'train.conll', ['words', 'pos', 'ignore', 'chunk']).iob_sents()
    test_sents = ConllCorpusReader('../data/vlsp2018', 'test.conll', ['words', 'pos', 'ignore', 'chunk']).iob_sents()

    train_sents = [x for x in train_sents if len(x) > 0]
    test_sents = [x for x in test_sents if len(x) > 0]

    print("#train_sents", len(train_sents))
    print("#test_sents", len(test_sents))
    return train_sents, test_sents


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 20.0, # 0: overfit , cao  --> underfit
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        # 'postag': postag,
        # 'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            # '-1:postag': postag1,
            # '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i > 1:
        word2 = sent[i-2][0]
        postag2 = sent[i-2][1]
        features.update({
            '-2:word.lower()': word2.lower(),
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isupper()': word2.isupper(),
            # '-2:postag': postag2,
            # '-2:postag[:2]': postag2[:2],
        })


    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            # '+1:postag': postag1,
            # '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


def train_crf(train_sents, saved=None):
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
        verbose=True
    )
    crf.fit(X_train, y_train)
    if saved is not None:
        pickle.dump(crf, open(saved, 'wb'))
    return crf


def evaluate(crf, test_sents):
    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]
    y_pred = crf.predict(X_test)

    labels = list(crf.classes_)
    labels.remove('O')
    metrics.flat_f1_score(y_test, y_pred,
                          average='weighted', labels=labels)

    # group B and I results
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))



if __name__ == '__main__':
    # Train conll 2003 (NER English dataset)
    print("Train vlsp 2003 (NER Vietnamese dataset)")
    saved = "./models/ner_vslp2018.pickle"
    train_sents, test_sents = load_data_vslp2018()
    crf = train_crf(train_sents, saved)



