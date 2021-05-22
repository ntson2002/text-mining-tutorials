# step 5
docs_new = ['what is computer',
            'who is Newton',
            'when is the Tet holiday ?']

id2label  = {0: 'DESC', 1: 'LOC', 2: 'HUM', 3: 'ENTY', 4: 'ABBR', 5: 'NUM'}
import joblib
text_clf = joblib.load('trec6_model.pkl')
predicted = text_clf.predict(docs_new)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, id2label[category]))
