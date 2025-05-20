# ------------------------------------------------------------------------
# Script to recreate confusion plots contained in repository. 
# Needs to point to experiments folder included in raw experiment download.
# ------------------------------------------------------------------------
# Adaption by: Marius Bock and Maximilian Hopp
# E-Mail: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------

import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# postprocessing parameters
steps = ['single_step', 'multi_step']
sampling_methods = ['shuffling', 'sequential', 'balanced']
datasets = ['wear', 'wetlab']
attacks = ['wainakh-simple', 'wainakh-whitebox', 'ebi', 'iLRG', 'llbgAVG', 'gcd']
states = ['untrained', 'trained']
models = ['deepconvlstm', 'tinyhar']
clip_level = 1.5
noise_level = 0.1

def main(args):
    for step in steps:
        if step == 'single_step':
            types = ['no_ldp', 'noise_ldp', 'clipping_ldp', 'clipping_noise_ldp']
        elif step == 'multi_step':
            types = ['N_100_S_5', 'N_100_S_2', 'N_500_S_5']

        for type in types:
            for sampling in sampling_methods:
                for dataset in datasets:
                    for attack in attacks:
                        for state in states:
                            for model in models:
                                if type == 'no_ldp':
                                    if state == 'untrained':
                                        path = '/{}/{}/{}/{}/{}/{}/seed_1/labels_{}_{}.csv'.format(dataset, model, state, step, type, sampling, model, attack)
                                    elif state == 'trained':
                                        path = os.path.join(args.experiment_folder, '{}/{}/{}/{}/{}/{}/seed_1/labelsT_{}_{}.csv'.format(dataset, model, state, step, type, sampling, model, attack))
                                elif type == 'noise_ldp':
                                    if state == 'untrained':
                                        path = os.path.join(args.experiment_folder, '{}/{}/{}/{}/{}/{}/{}/seed_1/labels_{}_{}.csv'.format(dataset, model, state, step, type, sampling, noise_level, model, attack))
                                    elif state == 'trained':
                                        path = os.path.join(args.experiment_folder, '{}/{}/{}/{}/{}/{}/{}/seed_1/labelsT_{}_{}.csv'.format(dataset, model, state, step, type, sampling, noise_level, model, attack))
                                elif type == 'clipping_ldp':
                                    if state == 'untrained':
                                        path = os.path.join(args.experiment_folder, '{}/{}/{}/{}/{}/{}/{}/seed_1/labels_{}_{}.csv'.format(dataset, model, state, step, type, sampling, clip_level, model, attack))
                                    elif state == 'trained':   
                                        path = os.path.join(args.experiment_folder, '{}/{}/{}/{}/{}/{}/{}/seed_1/labelsT_{}_{}.csv'.format(dataset, model, state, step, type, sampling, clip_level, model, attack))
                                elif type == 'clipping_noise_ldp':
                                    if state == 'untrained':
                                        path = os.path.join(args.experiment_folder, '{}/{}/{}/{}/{}/{}/{}/seed_1/labels_{}_{}.csv'.format(dataset, model, state, step, type, sampling, f'{noise_level}_{clip_level}', model, attack))
                                    elif state == 'trained':   
                                        path = os.path.join(args.experiment_folder, '{}/{}/{}/{}/{}/{}/{}/seed_1/labelsT_{}_{}.csv'.format(dataset, model, state, step, type, sampling, f'{noise_level}_{clip_level}', model, attack))

                                print('Processing:' + path)
                                if 'wetlab' in path:
                                    num_classes = 9
                                    sampling_rate = 50
                                    label_dict = {
                                        "null": 0,
                                        "cutting": 1,
                                        "inverting": 2,
                                        "peeling": 3,
                                        "pestling": 4,
                                        "pipetting": 5,
                                        "pouring": 6,
                                        "stirring": 7,
                                        "transfer": 8,
                                    }
                                elif 'wear' in path:
                                    num_classes = 19
                                    sampling_rate = 50
                                    label_dict = {
                                        'null': 0,
                                        'jogging': 1,
                                        'jogging (rotating arms)': 2,
                                        'jogging (skipping)': 3,
                                        'jogging (sidesteps)': 4,
                                        'jogging (butt-kicks)': 5,
                                        'stretching (triceps)': 6,
                                        'stretching (lunging)': 7,
                                        'stretching (shoulders)': 8,
                                        'stretching (hamstrings)': 9,
                                        'stretching (lumbar rotation)': 10,
                                        'push-ups': 11,
                                        'push-ups (complex)': 12,
                                        'sit-ups': 13,
                                        'sit-ups (complex)': 14,
                                        'burpees': 15,
                                        'lunges': 16,
                                        'lunges (complex)': 17,
                                        'bench-dips': 18
                                    }
                                # Load the predictions
                                data = pd.read_csv(path, header=0)

                                # Convert the labels to integers
                                all_gt = data.index.to_numpy()
                                all_preds = data['GT'].to_numpy()

                                # print label count in the ground truth
                                print("Ground truth label count:")
                                for i in range(num_classes):
                                    print(f"{i}: {np.sum(all_gt == i)}")
                                                                
                                comb_conf = confusion_matrix(all_gt, all_preds, normalize='true', labels=range(num_classes))
                                comb_conf = np.around(comb_conf, 2)
                                comb_conf[comb_conf == 0] = np.nan

                                _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
                                # create heatmap which is normalized between 0 and 1
                                sns.heatmap(comb_conf, annot=True, fmt='g', ax=ax, cmap=plt.cm.Greens, cbar=False, annot_kws={
                                            'fontsize': 16,}, linecolor='black', vmin=0, vmax=1)
                                ax.set_title(f'Confusion Matrix for {dataset} - {model} - {state} - {step} - {type} - {sampling} - {attack}', fontsize=20)
                                ax.set_xlabel('Predicted', fontsize=20)
                                ax.set_ylabel('True', fontsize=20)
                                ax.set_xticklabels(label_dict.keys(), fontsize=16)
                                ax.set_yticklabels(label_dict.keys(), fontsize=16)
                                plt.xticks(rotation=90)
                                plt.yticks(rotation=0)
                                """
                                ax.set_xticks([])
                                ax.set_yticks([])
                                """
                                # save the figure
                                # check if the directory exists, if not create it
                                if not os.path.exists(f'confusion_matrices/{dataset}/{model}/{step}/{attack}'):
                                    os.makedirs(f'confusion_matrices/{dataset}/{model}/{step}/{attack}')
                                
                                if type == 'no_ldp':
                                    file_name = f'{sampling}_{state}_{type}.png'
                                elif type == 'noise_ldp':
                                    file_name = f'{sampling}_{state}_{type}_{noise_level}.png'
                                elif type == 'clipping_ldp':
                                    file_name = f'{sampling}_{state}_{type}_{noise_level}.png'
                                elif type == 'clipping_noise_ldp':
                                    file_name = f'{sampling}_{state}_{type}_{noise_level}_{clip_level}.png'
                                elif type == 'N_100_S_5':
                                    file_name = f'{sampling}_{state}_{type}.png'
                                elif type == 'N_100_S_2':
                                    file_name = f'{sampling}_{state}_{type}.png'
                                elif type == 'N_500_S_5':
                                    file_name = f'{sampling}_{state}_{type}.png'
                        
                                save_path = f'confusion_matrices/{dataset}/{model}/{step}/{attack}'
                                plt.savefig(os.path.join(save_path, file_name), dpi=300)
                                plt.close()
                            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_folder', default='./experimental_results/raw')
    parser.add_argument('--save_folder', default='./experimental_results/confusion_matrices')
    
    args = parser.parse_args()
    
    main(args)  
    