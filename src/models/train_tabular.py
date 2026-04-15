from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def get_model(name):
    if name == "logreg":
        return LogisticRegression(max_iter=1000)
    return GradientBoostingClassifier()