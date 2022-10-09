import pickle
import time
from pathlib import Path

import pandas as pd

from scipy.sparse import hstack, coo_matrix, load_npz
from typing import Union, List, Any

from securityaware.core.models import get_model
from securityaware.handlers.plugin import PluginHandler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, make_scorer, confusion_matrix


class MLPipeline(PluginHandler):
    """
        ML pipeline plugin
    """

    class Meta:
        label = "ml_pipeline"

    def run(self, dataset: pd.DataFrame, model_type: str = 'RFC', training: bool = True, evaluate: bool = True,
            **kwargs) -> Union[pd.DataFrame, None]:
        """
            runs the plugin
        """
        train_features_path = self.get('train_features_path')
        train_labels_path = self.get('train_labels_path')
        test_features_path = self.get('test_features_path')
        test_labels_path = self.get('test_labels_path')
        train_data_path = self.get('train_data_path')
        test_data_path = self.get('test_data_path')

        # Set models path after data grouping options set.
        vectorizer = str(train_features_path.stem).split('_')[0]
        model, model_pipeline, model_prefix, model_param_grid = get_model(model_type,
                                                                          n_jobs=self.app.get_config('local_threads'))
        model_path = self.path / f"{self.output.stem}_nlp_{vectorizer}_{model_prefix}.model"
        self.set('model_path', model_path)
        predictions_path = self.path / f"{self.output.stem}_{model_prefix}_{vectorizer}_predictions.csv"
        # Load data.
        self.app.log.info(train_labels_path)
        train_labels = pd.read_csv(train_labels_path)
        test_labels = pd.read_csv(test_labels_path)
        y_train = train_labels["label"]
        y_test = test_labels["label"]

        # Create empty input.
        x_train = coo_matrix((y_train.shape[0], 0))
        x_test = coo_matrix((y_train.shape[0], 0))
        self.app.log.info(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")

        self.app.log.info("Adding NLP features.")
        train_features_sparse = load_npz(train_features_path)
        test_features_sparse = load_npz(test_features_path)
        x_train = hstack([train_labels, train_features_sparse])
        x_test = hstack([test_labels, test_features_sparse])
        self.app.log.info(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")

        # Simple train, test split.
        self.app.log.info(f"Train size: {x_train.shape}; Test size: {x_test.shape}")

        if training:
            # Tune, train and validate model.
            t_start = time.time()
            self.tune_train_and_validate(model_path=model_path, model=model_pipeline, param_grid=model_param_grid,
                                         x_train=x_train, y_train=y_train)
            train_time = time.time() - t_start

            self.app.log.info(f"Training time {train_time}")

        if evaluate:
            y_predict = self.evaluate_model(model_path=model_path, x_test=x_test, y_test=y_test)

            test_data = pd.read_csv(str(test_data_path))
            test_data['predicted'] = y_predict.tolist()
            test_data.to_csv(str(predictions_path), index=False)

        return None

    def tune_train_and_validate(self, model_path: Path, model: Any, param_grid: List[dict], x_train: Any, y_train: Any):
        # Perform gridsearch.
        self.app.log.info("\tPerforming grid search cross validation...")
        # Create the grid search cv object.
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=make_scorer(matthews_corrcoef), cv=3,
                                   n_jobs=self.app.get_config('local_threads'))

        # Perform grid search and cross validate.
        grid_search_results = grid_search.fit(x_train, y_train)

        # Print results.
        self.app.log.info("\tGrid search results:")
        for mean_score, params in zip(grid_search_results.cv_results_["mean_test_score"],
                                      grid_search_results.cv_results_["params"]):
            self.app.log.info(f"\t\t {mean_score} {params}")

        # Get the best model.
        best_model = grid_search_results.best_estimator_

        # Train the best model with the entire training set.
        self.app.log.info(f"\tTraining model with best hyper parameters {grid_search_results.best_params_}")
        best_model.fit(x_train, y_train)
        self.app.log.info("\tFinished training.")

        with model_path.open(mode='wb') as mf:
            pickle.dump(best_model, mf)

        self.app.log.info(f"\tSaved best model to {model_path}.")

    def evaluate_model(self, model_path: Path, x_test: Any, y_test: Any):
        model_best = pickle.load(open(model_path, 'rb'))

        p_start = time.time()
        y_predict = model_best.predict(x_test)
        pred_time = time.time() - p_start

        num_predictions = len([x for x in y_predict if x])
        num_predictions_p = num_predictions / x_test.shape[0]

        weighting = 'binary'
        mcc = matthews_corrcoef(y_test, y_predict)
        f1 = f1_score(y_test, y_predict, average=weighting)
        recall = recall_score(y_test, y_predict, average=weighting)
        precision = precision_score(y_test, y_predict, average=weighting)
        tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
        self.app.log.info(f"TN: {tn}; FP: {fp}; FN: {fn}; TP: {tp}")
        self.app.log.info(f"MCC: {mcc}, F1: {f1}, Recall: {recall}, Precision: {precision}, # Preds. {num_predictions_p}, Pred. time: {pred_time}")

#        rrfile.write(f"{active_project_name},{model_prefix},{models_sub_path},{bootstrap_run},{num_predictions},{num_predictions_p},{recall},{precision},{f1},{mcc},{train_time},{pred_time}\n")

        return y_predict


def load(app):
    app.handler.register(MLPipeline)
