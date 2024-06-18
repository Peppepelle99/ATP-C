import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

np.float = float





def load_dataset(split = None):
  data = np.load('../archives/Dataset_Liquid_Complete.npy')
  X = data[:, :-1]  
  y = data[:, -1]   


  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

  if split == 'TRAIN':
    return  X_train, y_train
  elif split == 'TEST':
    return X_test, y_test
  else:
    return X_train, y_train, X_test, y_test
  
def load_dataset_complete(split = None):
  data = np.load('../archives/Dataset_Complete.npy')
  X = data[:, :-1]  
  y = data[:, -1]   


  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

  if split == 'TRAIN':
    return  X_train, y_train
  elif split == 'TEST':
    return X_test, y_test
  else:
    return X_train, y_train, X_test, y_test


def load_dataset_platinum():

    data = np.load('archives/Dataset_Liquid_Platinum.npy')
    X = data[:, :-1]  
    y = data[:, -1]

    return X,y

def load_dataset_Copper():
    data = np.load('archives/Dataset_Liquid_Copper.npy')
    X = data[:, :-1]  
    y = data[:, -1]

    return X,y

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def downsampling(X, type):

    if type == 'simple':
        return X[:, ::5]
    
    if type == 'moving_avg':

        original_points = X.shape[1]
        target_points = 888
        window_size = int(original_points / target_points)  
        data_moving_avg = np.array([moving_average(sample, window_size) for sample in X])
        
        return data_moving_avg[:, ::10]
    
    if type == 'interpol':

        from scipy.interpolate import interp1d

        # Numero target di punti
        target_points = 1500

        # Creare nuovo array per i dati ridotti
        data_interpolated = np.zeros((X.shape[0], target_points))

        # Interpolare ciascun campione
        for i, sample in enumerate(X):
            x = np.linspace(0, len(sample)-1, len(sample))
            f = interp1d(x, sample, kind='linear')
            x_new = np.linspace(0, len(sample)-1, target_points)
            data_interpolated[i, :] = f(x_new)
        
        return data_interpolated



def load_dataset_condensatore(split = None):
    data = np.load('../archives/Dataset_Condensatore.npy')
    X = data[:, 1100:-1]  
    y = data[:, -1]   

    X = downsampling(X, 'moving_avg')

    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if split == 'TRAIN':
        return  X_train, y_train
    elif split == 'TEST':
        return X_test, y_test
    else:
        return X_train, y_train, X_test, y_test

def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path

def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def visualize_filter(root_dir):
    import tensorflow.keras as keras
    classifier = 'resnet'
    archive_name = 'UCRArchive_2018'
    dataset_name = 'GunPoint'
    datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    model = keras.models.load_model(
        root_dir + 'results/' + classifier + '/' + archive_name + '/' + dataset_name + '/best_model.hdf5')

    # filters
    filters = model.layers[1].get_weights()[0]

    new_input_layer = model.inputs
    new_output_layer = [model.layers[1].output]

    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

    classes = np.unique(y_train)

    colors = [(255 / 255, 160 / 255, 14 / 255), (181 / 255, 87 / 255, 181 / 255)]
    colors_conv = [(210 / 255, 0 / 255, 0 / 255), (27 / 255, 32 / 255, 101 / 255)]

    idx = 10
    idx_filter = 1

    filter = filters[:, 0, idx_filter]

    plt.figure(1)
    plt.plot(filter + 0.5, color='gray', label='filter')
    for c in classes:
        c_x_train = x_train[np.where(y_train == c)]
        convolved_filter_1 = new_feed_forward([c_x_train])[0]

        idx_c = int(c) - 1

        plt.plot(c_x_train[idx], color=colors[idx_c], label='class' + str(idx_c) + '-raw')
        plt.plot(convolved_filter_1[idx, :, idx_filter], color=colors_conv[idx_c], label='class' + str(idx_c) + '-conv')
        plt.legend()

    plt.savefig(root_dir + 'convolution-' + dataset_name + '.pdf')

    return 1

def plot_confusion_matrix(output_directory, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    # Genera un grafico della matrice di confusione utilizzando seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(3), yticklabels=range(3))
    plt.xlabel('Etichetta Predetta')
    plt.ylabel('Etichetta Vera')
    plt.title('Matrice di Confusione')
    plt.show()
    plt.savefig(output_directory+'/cm.png')

def plot_accuracy(output_directory, scores, stds, names, dataset):
   
    import matplotlib.pyplot as plt
 

    # Creazione del grafico a dispersione
    plt.scatter(names, scores, s=100*np.sqrt(stds), alpha=0.5)
    plt.title(f'Datset: {dataset}')
    plt.xlabel('Modelli')
    plt.ylabel('Accuratezza Media')
    plt.grid(True)

    # Aggiunta delle barre di errore
    for x, y,std in zip(names, scores, stds):
        plt.errorbar(x, y, yerr=std, fmt='o', capsize=5, color='black')

    plt.yticks(np.arange(0.65, 1, 0.05))


    plt.savefig(f'{output_directory}accuracy_plot_2_{dataset}.png')

def kfold_split(X, y, train_index, test_index, normalization=True ):
    x_train = X[train_index]
    y_train = y[train_index]
    x_test = X[test_index]
    y_test = y[test_index]

    #Z-score Normalization
    if normalization:
        
        std_ = x_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

    return x_train, y_train, x_test, y_test



def select_params(classifier_name, dataset = 'Liquid'):

    if classifier_name == 'resnet': 
        return {
                    'learning_rate': 0.001,
                    'mini_batch': 15,
                    'transfer_learning': False,
                    'num_epochs': 20
                }
    
    elif classifier_name == 'multiHydra': 
        if dataset == 'Liquid': #best 95% 
            return {
                        'n_kernels': 8,
                        'n_groups': 64
                    }
        elif dataset == 'Complete' :
           return { #best 81%
                        'n_kernels': 46,
                        'n_groups': 91
                    }
        else:
            return { #best 95
                        "n_kernels": 5,
                        "n_groups": 36
                    }
           
        
    elif classifier_name == 'inceptionT': 
        if dataset == 'Liquid': #best 93%
            return {
                    "batch_size": 64,
                    "num_epochs": 250,
                    "depth": 2,
                    "n_classifiers": 3
                }
        elif dataset == 'Complete': #best 83%
             return {
                    "batch_size": 34,
                    "num_epochs": 250,
                    "depth": 5,
                    "n_classifiers": 2
                }
        else:
            return { #best 92%
                    "batch_size": 44,
                    "num_epochs": 250,
                    "depth": 4,
                    "n_classifiers": 2
                } 
            
    
    elif classifier_name == 'rdst': 
        if dataset == 'Liquid': #best 93% 
            return {
                        'max_shapelets': 1000, 
                        'shapelet_lengths': 5,
                    }
        elif dataset == 'Complete':
           return { #best 77%
                        'max_shapelets': 1000, 
                        'shapelet_lengths': 11,
                    }
        else:
            return { #best 94
                        'max_shapelets': 1000, 
                        'shapelet_lengths': None,
                    }
        
    elif classifier_name == 'weasel-d': 
       
       if dataset == 'Liquid':#best 93%
            return {
                            'min_window': 4, 
                            'word_lengths': [3,4],
                        }
       elif dataset == 'Complete':
            return { #best 79%
                            'min_window': 4, 
                            'word_lengths': [7,10],
                        }
       else:
           return { #best 93%
                            'min_window': 8, 
                            'word_lengths': [10,11],
                        }
       
    elif classifier_name == 'freshPrince': 
       
       if dataset == 'Liquid': #best 93%
            return {
                            'n_estimators': 15, 
                        }
       else:
           return {
                            'n_estimators': 15, 
                        }
           
    elif classifier_name == 'drCif': 
       
       if dataset == 'Liquid': #best 93% 
            return {
                        'n_estimators': 25, 
                    }
       elif dataset == 'Complete':
            return { #best 81%
                        'n_estimators': 225, 
                    }
       else:
            return {
                        'n_estimators': 41, 
                    }
    else:

        return { 'none': None}