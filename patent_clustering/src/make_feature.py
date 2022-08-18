from sklearn.feature_extraction.text import TfidfVectorizer


def make_feature(text_data, modes=["tfidf"]):
    ### create features
    features_dict = {}
    # lets try tfidf
    if "tfidf" in modes:
        corpus = [
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ]

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        print(vectorizer.get_feature_names())
        print(len(vectorizer.get_feature_names()))
        print(X.toarray())
        print(X.toarray().shape)
        features_dict["tfidf"] = X.toarray()
    # lets try word2bec
    elif "word2vec" in modes:
        pass

    return features_dict


if __name__ == "__main__":
    make_feature(text_data=None)
