from ner_crf.vlsp2018_train import evaluate, load_data_vslp2018, sent2features
import pickle


if __name__ == '__main__':
    # Evaluate
    print("====================Load model and evaluate in the test data =========================")
    saved = "./models/ner_vslp2018.pickle"
    train_sents , test_sents = load_data_vslp2018()
    print(test_sents[0])
    crf = pickle.load(open(saved, 'rb'))
    evaluate(crf, test_sents)
    evaluate(crf, train_sents)

    print("==================== Predict from text ====================")
    print("=== Input text  =====")
    text = "Bà Hà là một trong hai hộ nuôi bò sữa đầu tiên của xã Hoà Bình."

    # from vncorenlp import VnCoreNLP
    # import itertools
    # tokenizer = VnCoreNLP(address='http://127.0.0.1', port=9000)  # elastic tokenizer
    # pos_tokens = tokenizer.pos_tag(text)
    # pos_tokens = list(itertools.chain(*pos_tokens))

    from underthesea import pos_tag
    pos_tokens = pos_tag(text)
    print("=== Tokenizing and POS  =====")
    print(pos_tokens)

    a_sentence = [i + ('-',) for i in pos_tokens]
    print("=== Conll format =====")
    for x in a_sentence:
        print(x)
    print("=== Feature      =====")
    a_sentence_feature = sent2features(a_sentence)
    print(a_sentence_feature)
    yyy = crf.predict([a_sentence_feature])[0]
    print("=== Output       =====")
    print(yyy)
    print("=== Output Detail===== ")
    for a, b in zip(a_sentence, yyy):
        print(a, "\t", b)

