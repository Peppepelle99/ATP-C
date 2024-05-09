import sys
from utils.utils import load_dataset, select_params, load_dataset_complete, create_directory, plot_accuracy
from utils.train_test import fit_classifier, test_classifier
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


mode = 'TRAIN'
ensamble = True

if ensamble:
    classifier_name = sys.argv[1:]

    print('Method: ', classifier_name, mode)
else:
    classifier_name = sys.argv[1]
    print('Method: ', classifier_name, mode)



if ensamble:
    #param_grid = select_params(classifier_name[0])

    param_grid = []
    for c in classifier_name:
        param_grid.append(select_params(c))
else:
    param_grid = select_params(classifier_name)

optimized_params = nni.get_next_parameter()
#param_grid.update(optimized_params)


if mode == 'TRAIN':

    dataset = load_dataset(split='TRAIN')
    mean_acc, std_acc = fit_classifier(dataset, param_grid, classifier_name)

    nni.report_final_result(mean_acc)

elif mode == 'TEST':
    output_dir = '../results/' + classifier_name
    create_directory(output_dir)

    test_classifier(param_grid, classifier_name, output_dir)

elif mode == 'accuracy_plot':
    output_dir = '../results/'
    create_directory(output_dir)

    dataset = load_dataset(split='TRAIN')

    names = ['hivecote2','multiHydra', 'rdst', 'inceptionT']

    all_scores = []
    all_std = []
    for name in names:
        param_grid = select_params(name)
        mean_acc, std_acc = fit_classifier(dataset, param_grid, name)
        all_scores.append(mean_acc)
        all_std.append(std_acc)

    plot_accuracy(output_dir, all_scores, all_std, names)



print('DONE')
