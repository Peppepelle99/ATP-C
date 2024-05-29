import sys
from utils.utils import load_dataset, select_params, load_dataset_complete, create_directory, plot_accuracy, plot_confusion_matrix
from utils.train_test import fit_classifier, test_classifier
import nni

# remove info-warning

import tensorflow as tf
import warnings
import logging
import os
import numpy as np

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
dataset_name = 'Complete'

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
    param_grid = select_params(classifier_name, dataset_name)

    optimized_params = nni.get_next_parameter()
    param_grid.update(optimized_params)


if mode == 'TRAIN':

    dataset = datasets[dataset_name](split='TRAIN')
    mean_acc, std_acc = fit_classifier(dataset, param_grid, classifier_name)

    nni.report_final_result(mean_acc)

elif mode == 'TEST':

    compare = True

    output_dir = '../results/'
    create_directory(output_dir)
    dataset = datasets[dataset_name]()

    if compare:
        names = ['rdst','multiHydra']
        all_pred = []
        all_true = []
        all_accuracy = []
        for name in names:
            param_grid = select_params(name)
            y_pred, y_true, acc = test_classifier(dataset, param_grid, name, output_dir)
            all_pred.append(np.array(y_pred))
            all_true.append(np.array(y_true))
            all_accuracy.append((name, acc))

        all_pred = np.concatenate(all_pred)
        all_true = np.concatenate(all_true)
        plot_confusion_matrix(output_dir, all_true, all_pred)
        print(all_accuracy)
    else:
        output_dir = output_dir + classifier_name
        test_classifier(dataset, param_grid, classifier_name, output_dir)

elif mode == 'accuracy_plot':
    output_dir = '../results/'
    create_directory(output_dir)

    dataset = datasets[dataset_name](split='TRAIN')

    names = ['rdst','multiHydra', 'inceptionT', 'weasel-d', 'hivecote2', 'freshPrince', 'drCif']

    all_scores = []
    all_std = []
    for name in names:
        param_grid = select_params(name, dataset_name)
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
