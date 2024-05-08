import sys
from utils.utils import load_dataset, select_params, load_dataset_complete, create_directory, plot_boxplot
from utils.train_test import fit_classifier, test_classifier, fit_ensamble
import nni

# remove info-warning

import tensorflow as tf
import warnings
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
tf.get_logger().setLevel('ERROR')  
warnings.filterwarnings("ignore")  
logging.getLogger('tensorflow').disabled = True 


mode = 'BoxPlot'
ensamble = False

if ensamble:
    classifiers= sys.argv[1:]

    print('Method: ', classifiers, mode)
else:
    classifier_name = sys.argv[1]
    print('Method: ', classifier_name, mode)



if ensamble:
    param_grid = select_params(classifiers[0])

    params = []
    for c in classifiers:
        params.append(select_params(c))
else:
    param_grid = select_params(classifier_name)

optimized_params = nni.get_next_parameter()
param_grid.update(optimized_params)


if mode == 'TRAIN':

    dataset = load_dataset(split='TRAIN')
    if ensamble:
        mean_acc, std_acc = fit_ensamble(dataset, params, classifiers)
    else:
        mean_acc, std_acc, scores = fit_classifier(dataset, param_grid, classifier_name)

    nni.report_final_result(mean_acc)

elif mode == 'TEST':
    output_dir = '../results/' + classifier_name
    create_directory(output_dir)

    test_classifier(param_grid, classifier_name, output_dir)

elif mode == 'BoxPlot':
    output_dir = '../results/'
    create_directory(output_dir)

    dataset = load_dataset(split='TRAIN')

    names = ['multiHydra', 'rdst']

    all_scores = []
    for name in names:
        param_grid = select_params(name)
        mean_acc, std_acc, scores = fit_classifier(dataset, param_grid, name)
        all_scores.append(scores)

    plot_boxplot(output_dir, all_scores, names)



print('DONE')
