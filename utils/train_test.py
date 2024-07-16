import numpy as np
from utils.utils import kfold_split, plot_confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import nni
import time

def fit_classifier(dataset, params, classifier_name):

    X, y = dataset

    print(params)
    

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = []        

    for i, (train_index, val_index) in enumerate(kfold.split(X, y)):
      

      x_train, y_train, x_val, y_val = kfold_split(X, y, train_index, val_index)

      x_train = x_train.reshape((x_train.shape[0],1, x_train.shape[1]))
      x_val = x_val.reshape((x_val.shape[0],1, x_val.shape[1]))     

      classifier = create_classifier(classifier_name, params)
      classifier.fit(x_train, y_train)
      y_pred = classifier.predict(x_val)

      acc = accuracy_score(y_val, y_pred)
      print(f'fold: {i}, accuracy = {acc}')
      scores.append(acc)
      nni.report_intermediate_result(acc)
    
    print(f'accuracy mean: {np.mean(scores)}, std: {np.std(scores)} \n\n')

    return np.mean(scores), np.std(scores)

def test_classifier(dataset, params, classifier_name, output_dir):

    if len(dataset) != 2:
        x_train, y_train, x_test, y_test = dataset
    else:
        x_train, y_train = dataset['train']
        x_test, y_test = dataset['test']
    

    print(params)

    std_ = x_train.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

    std_ = x_test.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_


    x_train = x_train.reshape((x_train.shape[0],1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0],1, x_test.shape[1]))     

    classifier = create_classifier(classifier_name, params)

    start_time_fit = time.time()
    classifier.fit(x_train, y_train)
    end_time_fit = time.time()

    start_time_pred = time.time()
    y_pred = classifier.predict(x_test)
    end_time_pred = time.time()

    
    plot_confusion_matrix(output_dir, y_test, y_pred)

    acc = accuracy_score(y_test, y_pred)
    print(f'test accuracy = {acc}')

    return y_pred, y_test, acc, round(end_time_pred-start_time_pred, 3), x_test.shape[0], round(end_time_fit-start_time_fit, 3), x_train.shape[0]

def create_classifier(classifier_name, params):
    resample_id = 1

    if classifier_name == 'hivecote2':
        from aeon.classification.hybrid import HIVECOTEV2
        return HIVECOTEV2(random_state = resample_id, time_limit_in_minutes=2)
    
    if classifier_name == 'multiHydra':
        from aeon.classification.convolution_based import MultiRocketHydraClassifier
        return MultiRocketHydraClassifier(n_kernels=params['n_kernels'], n_groups=params['n_groups'], random_state = resample_id)
    
    if classifier_name == 'inceptionT':
        from aeon.classification.deep_learning import InceptionTimeClassifier
        return InceptionTimeClassifier(n_epochs=params['num_epochs'],batch_size=params['batch_size'], n_classifiers = params['n_classifiers'], depth = params['depth'], verbose=False, random_state = resample_id)
    
    if classifier_name == 'rdst':
        s_l = [params['shapelet_lengths']] if params['shapelet_lengths'] != "None" else None
        from aeon.classification.shapelet_based import RDSTClassifier
        return RDSTClassifier(max_shapelets = params['max_shapelets'], shapelet_lengths = s_l, random_state = resample_id)
    
    if classifier_name == 'weasel-d':
        from aeon.classification.dictionary_based import WEASEL_V2
        return WEASEL_V2(min_window = params['min_window'], word_lengths = params['word_lengths'] ,random_state = resample_id)
    
    if classifier_name == 'freshPrince':
        from aeon.classification.feature_based import FreshPRINCEClassifier
        return FreshPRINCEClassifier(n_estimators=params['n_estimators'], random_state = resample_id)
    
    if classifier_name == 'drCif':
        from aeon.classification.interval_based import DrCIFClassifier
        return DrCIFClassifier(n_estimators=params['n_estimators'], random_state = resample_id)
    
    
    