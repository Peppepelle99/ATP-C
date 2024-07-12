import sys
from utils.utils import load_dataset, select_params, load_dataset_complete,load_dataset_condensatore, create_directory, plot_accuracy, plot_confusion_matrix, load_dataset_condensatore_isolante, load_dataset_condensatore_conduttivo, load_dataset_condensatore_miscela
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
    'Complete': load_dataset_complete,
    'Condensatore': load_dataset_condensatore,
    'Condensatore_isolante': load_dataset_condensatore_isolante,
    'Condensatore_conduttivo': load_dataset_condensatore_conduttivo,
    'Condensatore_miscela': load_dataset_condensatore_miscela
}

#SETTINGS
mode = 'TEST'
ensamble = False
dataset_name = 'Condensatore'

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
    param_grid = select_params(classifier_name, dataset=dataset_name)

    optimized_params = nni.get_next_parameter()
    param_grid.update(optimized_params)


if mode == 'TRAIN':

    dataset = datasets[dataset_name](split='TRAIN')
    mean_acc, std_acc = fit_classifier(dataset, param_grid, classifier_name)

    nni.report_final_result(mean_acc)

elif mode == 'TEST':

    compare = True

    output_dir = f'../results/{dataset_name}'

    # dataset = datasets[dataset_name]()
    
    dataset = {
        'train': datasets['Condensatore'](split='ALL'),
        'test': datasets['Condensatore_miscela'](split='ALL')
    }

    if len(dataset) == 2:
        X, y = dataset['test']
        print(f'class_-1: {X[y == -1].shape}')
        for x in range(16):
            print(f'class_{x}: {X[y == x].shape}')

    if compare:
        create_directory(output_dir)
        names = ['rdst','multiHydra', 'inceptionT', 'weasel-d', 'hivecote2', 'freshPrince', 'drCif']
        all_pred = []
        all_true = []
        all_accuracy = []
        for name in names:
            param_grid = select_params(name, dataset_name)
            y_pred, y_true, acc, prediction_time, test_size, train_time, train_size = test_classifier(dataset, param_grid, name, output_dir)
            all_pred.append(np.array(y_pred))
            all_true.append(np.array(y_true))
            all_accuracy.append((name, acc, prediction_time, train_time))

        print(f'train size: {train_size}, test size: {test_size}')
        all_pred = np.concatenate(all_pred)
        all_true = np.concatenate(all_true)
        plot_confusion_matrix(output_dir, all_true, all_pred)
        print(all_accuracy)
    else:
        output_dir = output_dir + '/' + classifier_name
        create_directory(output_dir)
        test_classifier(dataset, param_grid, classifier_name, output_dir)

elif mode == 'accuracy_plot':

    output_dir = f'../results/{dataset_name}'
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
