from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from utils.train_test import create_classifier

def ensamble(classifier_names, params):

    level_0 = []
    for name, param in zip(classifier_names, params):
       
       classifier = create_classifier(name, param)
       level_0.append((name,classifier))
    
    level_1 = LogisticRegression()

    model = StackingClassifier(estimators=level_0, final_estimator=level_1, cv=None)

    return model