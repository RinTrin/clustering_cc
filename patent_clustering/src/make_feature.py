from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import word2vec


def make_feature(opt, text_data, modes=["tfidf"]):
    ### create features
    features_dict = {}
    # lets try tfidf
    uuid_list = []
    corpus = []
    for idx, data_dict in text_data.items():
        uuid_list.append(data_dict["uuid"])
        corpus.append(data_dict["text"])
    if "tfidf" in modes:
        print("tfidf making feature start")
        # corpus = [
        #     "This is the first document.",
        #     "This document is the second document.",
        #     "And this is the third one.",
        #     "Is this the first document?",
        # ]

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        print(vectorizer.get_feature_names())
        print(len(vectorizer.get_feature_names()))
        print(X.toarray())
        print(X.toarray().shape)
        features_dict["tfidf"] = X.toarray()

        print("tfidf making feature finish")
    # lets try word2bec
    if "word2vec" in modes:

        model = word2vec.load_word2vec_format("path-to-vectors.txt", binary=False)
        vectors = [model[w] for w in corpus]
        features_dict["word2vec"] = vectors

    return features_dict


if __name__ == "__main__":
    make_feature(text_data=None)
