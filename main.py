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

datasets = {
    'Liquid': load_dataset,
    'Complete': load_dataset_complete
}

#SETTINGS
mode = 'accuracy_plot'
ensamble = False
dataset_name = 'Liquid'

if ensamble:
    classifier_name = sys.argv[1:]

    print('Method: ', classifier_name, mode)
else:
    classifier_name = sys.argv[1]
    print('Method: ', classifier_name, mode)

if ensamble:

    param_grid = []
    for c in classifier_name:
        param_grid.append(select_params(c))
else:
    param_grid = select_params(classifier_name)

    optimized_params = nni.get_next_parameter()
    param_grid.update(optimized_params)


if mode == 'TRAIN':

    dataset = datasets[dataset_name](split='TRAIN')
    mean_acc, std_acc = fit_classifier(dataset, param_grid, classifier_name)

    nni.report_final_result(mean_acc)

elif mode == 'TEST':

    output_dir = '../results/' + classifier_name
    create_directory(output_dir)
    dataset = datasets[dataset_name]()

    test_classifier(dataset, param_grid, classifier_name, output_dir)

elif mode == 'accuracy_plot':
    output_dir = '../results/'
    create_directory(output_dir)

    dataset = datasets[dataset_name](split='TRAIN')

    names = ['rdst','multiHydra']

    all_scores = []
    all_std = []
    for name in names:
        param_grid = select_params(name)
        mean_acc, std_acc = fit_classifier(dataset, param_grid, name)
        all_scores.append(mean_acc)
        all_std.append(std_acc)
    
    #sort in descend order
    sorted_indices = [i for i, _ in sorted(enumerate(all_scores), key=lambda x: x[1], reverse=True)]

    all_scores = [all_scores[i] for i in sorted_indices]
    all_std = [all_std[i] for i in sorted_indices]
    names = [names[i] for i in sorted_indices]

    plot_accuracy(output_dir, all_scores, all_std, names, dataset_name)



print('DONE')
