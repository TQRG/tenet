from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

hyperparameters = {
    'KNN': {
        '__weights': ['uniform', 'distance'],
        '__n_neighbors': [2, 3, 4, 5, 10, 20],
        '__metric': ['minkowski', 'euclidean', 'manhattan']
    },
    'SVC': {
        '__C': [0.01, 0.1, 1, 10],
        '__kernel': ['linear', 'poly', 'rbf'],
        '__gamma': [0.01, 0.1, 0.5, 1, 5]
    },
    'RFC': {
        '__max_depth': [10, 100, None],
        '__min_samples_leaf': [1, 2, 4],
        '__n_estimators': [200, 600, 1000]
    },
    'ADA': {
        '__n_estimators': [30, 50, 100],
        '__learning_rate': [0.01, 0.1, 1, 10]
    }
}


def get_model(model_type: str, n_jobs: int):
    model = None

    if model_type == "KNN":
        # KNN pipeline.
        model = KNeighborsClassifier(n_jobs=n_jobs)
    elif model_type == "SVC":
        # SVC pipeline.
        model = SVC(max_iter=200)
    elif model_type == "RFC":
        # Random forest pipeline.
        model = RandomForestClassifier(n_jobs=n_jobs)

    elif model_type == "ADA":
        # AdaBoost pipeline.
        model = AdaBoostClassifier()

    model_pipeline = make_pipeline(model)
    model_prefix = model.__class__.__name__.lower()

    # Parameter grid.
    model_param_grid = [{(model_prefix + k): v for k, v in hyperparameters[model_type].items()}]

    return model, model_pipeline, model_prefix, model_param_grid
