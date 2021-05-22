import codecs

def _generate_examples(filepath):
    examples = []
    with codecs.open(filepath, "rb") as f:
        for id_, row in enumerate(f):
            # One non-ASCII byte: sisterBADBYTEcity. We replace it with a space
            label, _, text = row.replace(b"\xf0",
                                         b" ").strip().decode().partition(" ")
            coarse_label, _, fine_label = label.partition(":")
            examples.append((id_, {
                "label-coarse": coarse_label,
                "label-fine": fine_label,
                "text": text,
            }))
    return examples


if __name__ == '__main__':
    train = _generate_examples("../data/trec_text_classification/train_5500.label")
    test = _generate_examples("../data/trec_text_classification/TREC_10.label")

    # step 1
    # lấy danh sách các nhãn trong dữ liệu huấn luyện
    labels = [x['label-coarse'] for _, x in train]
    set_labels = list(set(labels))
    label2id = {x: i for i, x in enumerate(set_labels)}
    id2label = {i: x for i, x in enumerate(set_labels)}

    print("------")
    print(len(labels))
    print("------")
    print(set_labels)
    print("------")
    print(label2id)
    print("------")
    print(id2label)

    # step2
    train_target = [label2id[x['label-coarse']] for _, x in train]
    train_data = [x['text'] for _, x in train]

    test_data = [x['text'] for _, x in test]
    test_target = [label2id[x['label-coarse']] for _, x in test]

    print("#training size", len(train))
    print("#testing size", len(test))
    print(train[0])
    print(train[1])
    print(test[0])
    print(test[1])
    print(train_data[0], train_target[0])
    print(train_data[1], train_target[1])


    # step 3
    from sklearn import svm
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    ngram_range = (1, 2)
    text_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=ngram_range)),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', svm.LinearSVC()),
    ])


    # step 4
    text_clf.fit(train_data, train_target)

    print(text_clf.get_params())

    # step 5
    docs_new = ['what is computer',
                'who is Newton',
                'when is the Tet holiday ?']

    print("id2label", id2label)
    predicted = text_clf.predict(docs_new)
    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, id2label[category]))

    import joblib
    joblib.dump([text_clf, id2label], 'models/trec6_model.pkl')
