# ------------------------------------------------------------------------
# Main script for conducting single-step leakage experiments. Script runs attacks directly without simulation. 
# ------------------------------------------------------------------------
# Adaption by: Marius Bock and Maximilian Hopp
# E-Mail: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------

import argparse
import torch
import torch.nn as nn
import torchvision
import breaching
import neptune
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import csv
import sys
from pprint import pprint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.os_utils import Logger, load_config
from utils.torch_utils import fix_random_seed
from models.DeepConvLSTM import DeepConvLSTM
from models.TinyHAR import TinyHAR
from utils.torch_utils import init_weights, worker_init_reset_seed, InertialDataset
from torch.utils.data import DataLoader
from Samplers import UnbalancedSampler, BalancedSampler
from Defense_Sampler import DefenseSampler
from DPrivacy import DPrivacy, BreachDP
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from neptune.types import File

import os
import random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Opt-in to the future behavior to prevent the warning
pd.set_option('future.no_silent_downcasting', True)

class data_config_inertial:
    modality = "vision"
    size = (1_281_167,)
    classes = 19
    shape = (1, 50, 12)
    normalize = True
    
    def __init__(self):
        self.attributes = {}  
    
    def __getitem__(self, key):
        return self.attributes.get(key)
    
    def __setitem__(self, key, value):
        self.attributes[key] = value

class Leakage():
    def __init__(self, args = None, run = None, config = None, clipping = None, noise = None):
        self.args = args
        self.run = run
        self.config = config
        self.dpri = DPrivacy(multiplier=0.1, clip=0.1)
        self.breachingDP = BreachDP(local_diff_privacy={"gradient_noise": noise, "input_noise": 0.0, "distribution": "gaussian", "per_example_clipping": clipping}, setup=dict(device=torch.device("cpu"), dtype=torch.float))
    
    def main(self, args):
        if args.neptune:
            run = neptune.init_run(
            project="wasedo/leakage",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiNzYwZjBiMi0xMzEwLTQyMGEtOTFkMC01M2JjMGQzNzc0OTUifQ==",
            )
        else:
            run = None

        config = load_config(args.config)
        config['init_rand_seed'] = args.seed
        config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if args.neptune:
            run_id = run["sys/id"].fetch()
        else:
            run_id = args.run_id
        
        log_dir = os.path.join('logs', config['name'], run_id)
        sys.stdout = Logger(os.path.join(log_dir, 'log.txt'))

        # save the current cfg
        with open(os.path.join(log_dir, 'cfg.txt'), 'w') as fid:
            pprint(config, stream=fid)
            fid.flush()
            
        config['loader']['train_batch_size'] = args.batch_size
        
        if args.neptune:
            run['config_name'] = args.config
            run['config'].upload(os.path.join(log_dir, 'cfg.txt'))
            run['args/trained'] = args.trained
            run['args/model'] = config['name']
            run['args/dataset'] = config['dataset_name']
            run['args/sampling'] = args.sampling
            run['args/datapoints'] = config['loader']['train_batch_size']
            run['args/classes'] = config['dataset']['num_classes']
            run['args/label_strat_array'] = args.label_strat_array
            run['args/noise'] = args.noise
            run['args/clipping'] = args.clipping
        
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        setup = dict(device=torch.device("cpu"), dtype=torch.float)
        
        trained = args.trained
        strat_array = args.label_strat_array
        grads = []

        for label_strat in strat_array:
            for i, anno_split in enumerate(config['anno_json']):
                with open(anno_split) as f:
                    file = json.load(f)
                anno_file = file['database']
                config['labels'] = ['null'] + list(file['label_dict'])
                config['label_dict'] = dict(zip(config['labels'], list(range(len(config['labels'])))))
                train_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Training']
                val_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']

                print('Split {} / {}'.format(i + 1, len(config['anno_json'])))
                config['dataset']['json_anno'] = anno_split
                    
            
                split_name = config['dataset']['json_anno'].split('/')[-1].split('.')[0]
                # load train and val inertial data
                train_data, val_data = np.empty((0, config['dataset']['input_dim'] + 2)), np.empty((0, config['dataset']['input_dim'] + 2))
                for t_sbj in train_sbjs:
                    t_data = pd.read_csv(os.path.join(config['dataset']['sens_folder'],  t_sbj + '.csv'), index_col=False, low_memory=False).replace({"label": config['label_dict']}).infer_objects(copy=False).fillna(0).to_numpy()
                    train_data = np.append(train_data, t_data, axis=0)
                for v_sbj in val_sbjs:
                    v_data = pd.read_csv(os.path.join(config['dataset']['sens_folder'],  v_sbj + '.csv'), index_col=False, low_memory=False).replace({"label": config['label_dict']}).infer_objects(copy=False).fillna(0).to_numpy()
                    val_data = np.append(val_data, v_data, axis=0)

                # define inertial datasets
                train_dataset = InertialDataset(train_data, config['dataset']['window_size'], config['dataset']['window_overlap'])
                test_dataset = InertialDataset(val_data, config['dataset']['window_size'], config['dataset']['window_overlap'])

                # define dataloaders
                unbalanced_sampler = UnbalancedSampler(test_dataset, random.randint(0, config['dataset']['num_classes']), random.randint(0, config['dataset']['num_classes']))
                balanced_sampler = BalancedSampler(test_dataset)
                defense_sampler = DefenseSampler(test_dataset, random.randint(0, config['dataset']['num_classes']))
                
                config['init_rand_seed'] = args.seed
                
                rng_generator = fix_random_seed(config['init_rand_seed'], include_cuda=True) 
                val_loader = DataLoader(test_dataset, config['loader']['train_batch_size'], shuffle=True, num_workers=1, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
                unbalanced_loader = DataLoader(test_dataset, config['loader']['train_batch_size'], sampler=unbalanced_sampler, num_workers=1, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
                balanced_loader = DataLoader(test_dataset, config['loader']['train_batch_size'], sampler=balanced_sampler, num_workers=1, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
                defense_loader = DataLoader(test_dataset, config['loader']['train_batch_size'], sampler=defense_sampler, num_workers=1, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
                seq_loader = DataLoader(test_dataset, config['loader']['train_batch_size'], shuffle=False, num_workers=1, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
                
                if 'tinyhar' in config['name'] and trained:
                    args.resume = 'saved_models/' + config['dataset_name'] + f'/tinyhar/epoch_100_loso_sbj_{i}.pth.tar'
                if 'deepconvlstm' in config['name'] and trained:
                    args.resume = 'saved_models/' + config['dataset_name'] + f'/deepconvlstm/epoch_100_loso_sbj_{i}.pth.tar'

                if 'deepconvlstm' in config['name']:
                    model = DeepConvLSTM(
                        config['dataset']['input_dim'], config['dataset']['num_classes'] + 1, train_dataset.window_size,
                        config['model']['conv_kernels'], config['model']['conv_kernel_size'], 
                        config['model']['lstm_units'], config['model']['lstm_layers'], config['model']['dropout']
                        )
                    print("Number of learnable parameters for DeepConvLSTM: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
                    
                    # define criterion and optimizer
                    opt = torch.optim.Adam(model.parameters(), lr=config['train_cfg']['lr'], weight_decay=config['train_cfg']['weight_decay'])
                    
                    if args.resume and trained:
                        if os.path.isfile(os.path.join(os.getcwd(), args.resume)):
                            checkpoint = torch.load(args.resume, map_location = device) # loc: storage.cuda(config['device'])
                        
                            model.load_state_dict(checkpoint['state_dict'])
                            opt.load_state_dict(checkpoint['optimizer'])
                            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(args.resume, checkpoint['epoch']))
                            del checkpoint
                        else:
                            print("=> no checkpoint found at '{}'".format(args.resume))
                            return
                    else:
                        model = init_weights(model, config['train_cfg']['weight_init'])
                        pass
                    
                if 'tinyhar' in config['name']:
                    model = TinyHAR((config['loader']['train_batch_size'], 1, train_dataset.window_size, config['dataset']['input_dim']), config['dataset']['num_classes'] + 1, 
                                    config['model']['conv_kernels'], 
                                    config['model']['conv_layers'], 
                                    config['model']['conv_kernel_size'], 
                                    dropout=config['model']['dropout'], feature_extract=config['model']['feature_extract'])
                    
                    if args.resume and trained:
                        if os.path.isfile(os.path.join(os.getcwd(), args.resume)):
                            checkpoint = torch.load(args.resume, map_location = device) # loc: storage.cuda(config['device'])
                        
                            model.load_state_dict(checkpoint['state_dict'])
                            #opt.load_state_dict(checkpoint['optimizer'])
                            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(args.resume, checkpoint['epoch']))
                            del checkpoint
                        else:
                            print("=> no checkpoint found at '{}'".format(args.resume))
                            return
                    else:
                        model = init_weights(model, config['train_cfg']['weight_init'])
                
                model.train()
                    
                loss_fn = torch.nn.CrossEntropyLoss()
                
                if args.neptune:
                    run['label_attack' + '/' + str(label_strat) + '/attack_config_name'] = args.attack
                    
                if trained:
                    with open(os.path.join(log_dir, 'labelsT_{}_{}.csv'.format(config['name'], label_strat)), 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['GT', 'Pred', 'Sbj'])
                    file.close()

                    with open(os.path.join(log_dir, 'gradientsT_{}_{}.csv'.format(config['name'], label_strat)), 'w', newline='') as file:
                        writer = csv.writer(file)
                        for g in range(config['dataset']['num_classes'] + 1):
                            grads.append('G' + str(g))
                        writer.writerow(grads)
                    file.close()
                else:
                    with open(os.path.join(log_dir, 'labels_{}_{}.csv'.format(config['name'], label_strat)), 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['GT', 'Pred', 'Sbj'])
                    file.close()

                    with open(os.path.join(log_dir, 'gradients_{}_{}.csv'.format(config['name'], label_strat)), 'w', newline='') as file:
                        writer = csv.writer(file)
                        for g in range(config['dataset']['num_classes'] + 1):
                            grads.append('G' + str(g))
                        writer.writerow(grads)
                    file.close()

                # This is the attacker:
                cfg_attack = breaching.get_attack_config(args.attack)
                cfg_attack['optim']['max_iterations'] = int(config['attack']['iterations'])
                cfg_attack['label_strategy'] = label_strat
                
                if args.neptune:
                    log_dir_atk = os.path.join('logs', args.attack, '_' + run_id)
                    sys.stdout = Logger(os.path.join(log_dir_atk, 'log.txt'))
                    with open(os.path.join(log_dir_atk, 'cfg_atk.txt'), 'w') as fid:
                        pprint(cfg_attack, stream=fid)
                        fid.flush()
                    
                    run['label_attack' + '/' + str(label_strat) + '/attack_config'].upload(os.path.join(log_dir_atk, 'cfg_atk.txt'))
                    
                attacker = breaching.attacks.prepare_attack(model, loss_fn, cfg_attack, setup)

                # ## Simulate an attacked FL protocol
                # Server-side computation:
                metadata = data_config_inertial()
                metadata.shape = (1, 50, config['dataset']['input_dim'])
                metadata.classes = config['dataset']['num_classes'] + 1
                metadata['task'] = 'classification'
                
                server_payload = [
                    dict(
                        parameters=[p for p in model.parameters()], buffers=[b for b in model.buffers()], metadata=metadata
                    )
                ]

                # LOAD DATA
                if args.sampling == 'shuffle':
                    loader = val_loader 
                elif args.sampling == 'balanced':
                    loader = balanced_loader    
                elif args.sampling == 'unbalanced':
                    loader = unbalanced_loader
                elif args.sampling == 'defense':
                    loader = defense_loader
                elif args.sampling == 'sequential':
                    loader = seq_loader
                
                recovered_labels_all = []
                batchLabels_all = []
                mean_ln = 0
                mean_le = 0
                for i, (inputs, targets) in enumerate(loader, 0):
                    if 'deepconvlstm' in config['name']:
                        val_data = inputs
                        labels = targets
                        batchLabels = targets
                        
                        if(val_data.shape[0] != config['loader']['train_batch_size'] or 
                        val_data.shape[1] != 50 or 
                        val_data.shape[2] != config['dataset']['input_dim']):
                            break

                        
                    if 'tinyhar' in config['name']:
                        val_data = inputs
                        labels = targets
                        batchLabels = targets
                        
                        if(val_data.shape[0] != config['loader']['train_batch_size'] or 
                        val_data.shape[1] != 50 or 
                        val_data.shape[2] != config['dataset']['input_dim']):
                            break

                    
                    if config['name'] == 'ResNet':
                        val_data = inputs
                        labels = targets
                        batchLabels = targets
                        onedimdata = val_data
                        
                        if(val_data.shape[0] != config['loader']['train_batch_size'] or 
                        val_data.shape[1] != 50 or 
                        val_data.shape[2] != config['dataset']['input_dim']):
                            break
                    
                    # Normalize data
                    onedimdata = val_data.unsqueeze(1)
                    onedimdata = (onedimdata - onedimdata.min()) / (onedimdata.max() - onedimdata.min())
                    
                    output = model(onedimdata)
                    loss = loss_fn(output, labels)
                    
                    unique_labels, counts = torch.unique(labels, return_counts=True)
                    label_counts = dict(zip(unique_labels.tolist(), counts.tolist()))

                     # Write label counts to CSV
                    with open(os.path.join(log_dir, 'label_counts.csv'), 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['Label', 'Count'])
                        for label, count in label_counts.items():
                            writer.writerow([label, count])
                    file.close()
                    
                    if run is not None:
                        run['label_attack' + '/' + str(label_strat) + '/' + split_name + "/clipping"] = self.breachingDP.clip_value
                        run['label_attack' + '/' + str(label_strat) + '/' + split_name + "/grad_noise"] = self.breachingDP.local_diff_privacy["gradient_noise"]
                        run['label_attack' + '/' + str(label_strat) + '/' + split_name + "/input_noise"] = self.breachingDP.local_diff_privacy["input_noise"]
                        run['label_attack' + '/' + str(label_strat) + '/' + split_name + "/distribution"] = self.breachingDP.local_diff_privacy["distribution"]

                    gradients = torch.autograd.grad(loss, model.parameters())
                    if args.clipping > 0:
                        gradients_clipped = self.breachingDP._clip_list_of_grad_([g.clone() for g in gradients])
                        gradients = gradients_clipped
                        
                    if args.noise > 0:
                        gradients_noise = self.breachingDP.applyNoise([g.clone() for g in gradients])
                        gradients = gradients_noise                    
                                    
                    shared_data = [
                        dict(
                            gradients=gradients,
                            buffers=None,
                            metadata=dict(num_data_points=config['loader']['train_batch_size'], labels=None, local_hyperparams=None,),
                        )
                    ]
                    
                    # Attack:
                    reconstructed_user_data, _ = attacker.reconstruct(server_payload, shared_data, {}, dryrun=False)
                    
                    recovered_labels = reconstructed_user_data['labels']
                    recovered_labels = recovered_labels.sort()[0]
                    batchLabels = batchLabels.sort()[0]

                    reconstructed_user_data = reconstructed_user_data['data']
                    reconstructed_user_data = (reconstructed_user_data - reconstructed_user_data.min()) / (reconstructed_user_data.max() - reconstructed_user_data.min())
                                        
                    gradient = torch.round(shared_data[0]["gradients"][-1], decimals=6)
                    
                    appendGradient = {f'G{i}': [gradient[i].item()] for i in range(config['dataset']['num_classes'] + 1)}
                    appendGradient = pd.DataFrame(appendGradient)
                    
                    appendData = pd.DataFrame({
                        'GT': batchLabels.numpy(),
                        'Pred': recovered_labels.numpy(),
                        'Sbj': split_name,
                        'idx': str(i)
                    })

                    # Append to CSV without writing the header again
                    if trained:
                        appendData.to_csv(os.path.join(log_dir, 'labelsT_' + str(config['name']) + '_' + str(label_strat) + '.csv'), mode='a', header=False, index=False, float_format='%.4f')
                        appendGradient.to_csv(os.path.join(log_dir, 'gradientsT_' + str(config['name']) + '_' + str(label_strat) + '.csv'), mode='a', header=False, index=False, float_format='%.4f')
                    else: 
                        appendData.to_csv(os.path.join(log_dir, 'labels_' + str(config['name']) + '_' + str(label_strat) + '.csv'), mode='a', header=False, index=False, float_format='%.4f')
                        appendGradient.to_csv(os.path.join(log_dir, 'gradients_' + str(config['name']) + '_' + str(label_strat) + '.csv'), mode='a', header=False, index=False, float_format='%.4f')
                        
                    correct = 0
                    wrong = 0
                    predicted_labels = recovered_labels.clone()

                    for label in batchLabels:
                        if label in predicted_labels:
                            correct += 1
                            index = np.where(predicted_labels == label)[0][0]

                            # Delete the first occurrence
                            predicted_labels = np.delete(predicted_labels, index)
                        else:
                            wrong +=1
                            
                    # Calculate Label Leakage Accuracy for label existence
                    unique_labelsGT = torch.unique(batchLabels)
                    unique_labelsPD = torch.unique(recovered_labels)
                    leAcc = 0
                    leAccWrong = 0  
                    for label in unique_labelsPD:
                        if label in unique_labelsGT:
                            leAcc += 1
                        elif label not in unique_labelsGT:
                            leAccWrong += 1
                    leAcc = leAcc / unique_labelsGT.size()[0] # Label Existence Accuracy
                    leAccWrong = leAccWrong / unique_labelsPD.size()[0] # Label Existence Prediction that were predicted, but not actually in the batch
                                        
                    lnAcc = (batchLabels.size()[0] - wrong) / batchLabels.size()[0]

                    block2 = ''
                    block2 += 'Correct Labels: ' + str(correct) + '\n' 
                    block2 += 'Wrong Labels: ' + str(wrong) + '\n' 
                    block2 += 'LnAcc: {:.2f}'.format(lnAcc * 100) + '\n'
                    block2 += '\n'

                    block1 = '\nLABEL LEAKAGE RESULTS:'
                    print('\n'.join([block1, block2]))
                    
                    # submit final values to neptune 
                    if run is not None:
                        run['label_attack' + '/' + str(label_strat) + '/' + split_name].append({"leAcc": leAcc})
                        run['label_attack' + '/' + str(label_strat) + '/' + split_name].append({"lnAcc": lnAcc})
                        
                    mean_ln += lnAcc
                    mean_le += leAcc
                    
                    recovered_labels_all.append(recovered_labels)
                    batchLabels_all.append(batchLabels) 
                    
                batchLabels_all = torch.cat(batchLabels_all)
                recovered_labels_all = torch.cat(recovered_labels_all)
                conf_mat = confusion_matrix(batchLabels_all, recovered_labels_all, normalize='true', labels=range(len(config['labels'])))
                classAvg_acc = conf_mat.diagonal().mean()
                
                # save final raw confusion matrix
                _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
                ax.set_title('Confusion Matrix: ' + str(label_strat) + ' ' + split_name)
                conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels']) 
                conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
                if run is not None:
                    run['label_attack' + '/' + str(label_strat) + '/' + split_name + '/conf_matrices'].append(File.as_image(plt.gcf()), name='all')
                plt.close()
                
                if run is not None:
                    run['label_attack' + '/' + str(label_strat) + '/' + split_name +'/final_lnAcc'] = mean_ln / len(loader)
                    run['label_attack' + '/' + str(label_strat) + '/' + split_name +'/final_leAcc'] = mean_le / len(loader)
                    run['label_attack' + '/' + str(label_strat) + '/' + split_name +'/final_classAvgAcc'] = classAvg_acc
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/leakage/wetlab_loso_deep.yaml')
    parser.add_argument('--eval_type', default='loso')
    parser.add_argument('--neptune', action='store_true', default=False)
    parser.add_argument('--run_id', default='run', type=str)
    parser.add_argument('--seed', default=1, type=int)       
    parser.add_argument('--gpu', default='cuda:0', type=str)
    
    # Leakage arguments
    parser.add_argument('--attack', default='_default_optimization_attack', type=str)
    parser.add_argument('--label_strat_array', nargs='+', default=['wainakh-simple', 'wainakh-whitebox', 'ebi', 'iLRG', 'llbgAVG', 'gcd'], type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--trained', action='store_true', default=False)
    parser.add_argument('--sampling', default='sequential', choices=['sequential', 'balanced', 'shuffle'], type=str)    
    parser.add_argument('--clipping', default=0.0, type=float)                                                     
    parser.add_argument('--noise', default=0.0, type=float)  
    
    args = parser.parse_args()
    
    leakage = Leakage(clipping=args.clipping, noise=args.noise)  
    leakage.main(args)
