# Arda Mavi
from sklearn.externals import joblib

def getClassifier(dir):
    # Getting trained classifier:
    try:
        clf = joblib.load(dir)
    except:
        return None
    return clf

def trainClassifier(clf, X, y):
    # Training classifier:
    return clf.fit(X, y)

def getScore(clf, X, y):
    # Get score:
    return clf.score(X, y)

def getPredict(clf, img):
    # Get predict:
    return clf.predict(img)

def saveClassifier(clf, dir):
    # Save classifier:
    joblib.dump(clf, dir)
