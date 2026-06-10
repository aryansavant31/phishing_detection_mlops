from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, 
                              RandomForestClassifier)

MODELS = {
    "logistic_regression": LogisticRegression(verbose=1),
    "k_neighbors": KNeighborsClassifier(),
    "decision_tree": DecisionTreeClassifier(),
    "ada_boost": AdaBoostClassifier(),
    "gradient_boosting": GradientBoostingClassifier(verbose=1),
    "random_forest": RandomForestClassifier(verbose=1)
}
