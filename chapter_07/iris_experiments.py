from joblib import parallel_backend
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


MODELS = [
    NearestCentroid(),
    KNeighborsClassifier(n_neighbors=3),
    GaussianNB(),
    MultinomialNB(),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=5),
    SVC(kernel="linear", C=1.0),
    SVC(kernel="rbf", C=1.0, gamma=0.25),
    SVC(kernel="rbf", C=1.0, gamma=0.001),
    SVC(kernel="rbf", C=1.0, gamma=0.001),
]


def run(x_train, y_train, x_test, y_test, clf):
    with parallel_backend("threading", n_jobs=-1):
        clf.fit(x_train, y_train)
        print(f"\tpredictions  :{clf.predict(x_test)}")
        print(f"\tactual labels:{y_test}")
        print(f"\tscore        :{clf.score(x_test, y_test):.4f}")


def main():
    x = np.load("../data/iris/iris_features.npy")
    y = np.load("../data/iris/iris_labels.npy")
    N = 120
    x_train = x[:N]
    x_test = x[N:]
    y_train = y[:N]
    y_test = y[N:]
    xa_train = np.load("../data/iris/iris_train_features_augmented.npy")
    ya_train = np.load("../data/iris/iris_train_labels_augmented.npy")
    xa_test = np.load("../data/iris/iris_test_features_augmented.npy")
    ya_test = np.load("../data/iris/iris_test_labels_augmented.npy")
    for model in MODELS[:5]:
        print(f"{model}")
        run(x_train, y_train, x_test, y_test, model)
    print("Starting augmented data:\n")
    for model in MODELS[5:]:
        print(f"{model}:")
        run(xa_train, ya_train, xa_test, ya_test, model)


if __name__ == "__main__":
    main()
