import sys
# from sklearn.feature_extraction import TfidfVectorizer
# sklearn 0.13 버전 이하에서는 이렇게 사용하라
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.externals import joblib
import joblib
from sklearn.metrics import confusion_matrix

movie_reviews_data_folder = 'txt_sentoken'
dataset = load_files(movie_reviews_data_folder, shuffle=False)
# print("n_samples: %d" % len(dataset.data))

docs_train, docs_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.5, random_state=None)
pipeline = Pipeline([
    ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
    ('clf', LinearSVC(C=1)),
])
parameters = {'vect__ngram_range': [(1, 1), (2, 2), (3, 3)]}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=1)

grid_search.fit(docs_train, y_train)

best_parameters = grid_search.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
    vect__ngram_range: (1, 2)

best_model = grid_search.best_estimator_

joblib.dump(grid_search.best_estimator_, 'model.file', compress=1)

y_hat_test = best_model.predict(docs_test)

# confusion_matrix(y_test, y_train)

