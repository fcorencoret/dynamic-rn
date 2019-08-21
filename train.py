"""""""""
Pytorch implementation of "A simple neural network module for relational reasoning"
"""""""""
from __future__ import print_function

import argparse
import json
import os
import pickle
import re
import numpy as np
from comet_ml import Experiment, ExistingExperiment
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm, trange

import utils
import math
import random
from clevr_dataset_connector import ClevrDataset, ClevrDatasetStateDescription
from model import RN

import pdb


def train(data, model, optimizer, epoch, args):
    model.train()

    avg_loss = 0.0
    n_batches = 0
    corrects = 0
    n_samples = 0
    progress_bar = tqdm(data)
    for batch_idx, sample_batched in enumerate(progress_bar):
        img, qst, label = utils.load_tensor_data(sample_batched, args.cuda, args.invert_questions)

        # forward and backward pass
        optimizer.zero_grad()
        output, l1_reg = model(img, qst)
        pred = output.data.max(1)[1]
        loss = F.nll_loss(output, label) + args.l1_lambd * l1_reg.mean().item()
        loss.backward()

        # compute global accuracy
        corrects += (pred == label.data).sum()
        # assert corrects == sum(class_corrects.values()), 'Number of correct answers assertion error!'
        # invalids = sum(class_invalids.values())
        n_samples += len(label)
        accuracy = corrects.item() / n_samples
        # assert n_samples == sum(class_n_samples.values()), 'Number of total answers assertion error!'

        # Gradient Clipping
        if args.clip_norm:
            clip_grad_norm_(model.parameters(), args.clip_norm)

        optimizer.step()

        # Show progress
        progress_bar.set_postfix(dict(loss='{:.4}'.format(loss.item()), acc='{:.2%}'.format(accuracy)))
        avg_loss += loss.item()
        n_batches += 1

        if batch_idx % args.log_interval == 0:
            avg_loss /= n_batches
            processed = batch_idx * args.batch_size
            total_n_samples = len(data) * args.batch_size
            progress = float(processed) / total_n_samples
            print('Train Epoch: {} [{}/{} ({:.0%})] Train loss: {:.4} Train Accuracy: {:.2%}'.format(
                epoch, processed, total_n_samples, progress, avg_loss, accuracy))
            avg_loss = 0.0
            n_batches = 0
    torch.cuda.empty_cache()
    return accuracy


def test(data, model, epoch, dictionaries, args):
    model.eval()

    # accuracy for every class
    class_corrects = {}
    # for every class, among all the wrong answers, how much are non pertinent
    class_invalids = {}
    # total number of samples for every class
    class_n_samples = {}
    # initialization
    for c in dictionaries[2].values():
        class_corrects[c] = 0.0
        class_invalids[c] = 0.0
        class_n_samples[c] = 0.0

    corrects = 0.0
    invalids = 0.0
    n_samples = 0

    inverted_answ_dict = {v: k for k,v in dictionaries[1].items()}
    sorted_classes = sorted(dictionaries[2].items(), key=lambda x: hash(x[1]) if x[1]!='number' else int(inverted_answ_dict[x[0]]))
    sorted_classes = [c[0]-1 for c in sorted_classes]

    confusion_matrix_target = []
    confusion_matrix_pred = []

    sorted_labels = sorted(dictionaries[1].items(), key=lambda x: x[1])
    sorted_labels = [c[0] for c in sorted_labels]
    sorted_labels = [sorted_labels[c] for c in sorted_classes]

    avg_loss = 0.0
    progress_bar = tqdm(data)
    # with torch.no_grad():
    for batch_idx, sample_batched in enumerate(progress_bar):
        img, qst, label = utils.load_tensor_data(sample_batched, args.cuda, args.invert_questions, volatile=True)
            
        output, l1_reg = model(img, qst)
        pred = output.data.max(1)[1]

        loss = F.nll_loss(output, label) + args.l1_lambd * l1_reg.mean().item()

        # compute per-class accuracy
        pred_class = [dictionaries[2][o.item()+1] for o in pred]
        real_class = [dictionaries[2][o.item()+1] for o in label.data]
        for idx,rc in enumerate(real_class):
            class_corrects[rc] += (pred[idx] == label.data[idx]).item()
            class_n_samples[rc] += 1

        for pc, rc in zip(pred_class,real_class):
            class_invalids[rc] += (pc != rc)

        for p,l in zip(pred, label.data):
            confusion_matrix_target.append(sorted_classes.index(l))
            confusion_matrix_pred.append(sorted_classes.index(p))
            
        # compute global accuracy
        corrects += (pred == label.data).sum().item()
        assert corrects == sum(class_corrects.values()), 'Number of correct answers assertion error!'
        invalids = sum(class_invalids.values())
        n_samples += len(label)
        assert n_samples == sum(class_n_samples.values()), 'Number of total answers assertion error!'
            
        avg_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            accuracy = corrects / n_samples
            invalids_perc = invalids / n_samples
            progress_bar.set_postfix(dict(acc='{:.2%}'.format(accuracy), inv='{:.2%}'.format(invalids_perc)))
    
    avg_loss /= len(data)
    invalids_perc = invalids / n_samples      
    global_accuracy = corrects / n_samples

    print('Test Epoch {}: Accuracy = {:.2%} ({:g}/{}); Invalids = {:.2%} ({:g}/{}); Test loss = {}'.format(epoch, accuracy, corrects, n_samples, invalids_perc, invalids, n_samples, avg_loss))
    for v in class_n_samples.keys():
        accuracy = 0
        invalid = 0
        if class_n_samples[v] != 0:
            accuracy = class_corrects[v] / class_n_samples[v]
            invalid = class_invalids[v] / class_n_samples[v]
        print('{} -- acc: {:.2%} ({}/{}); invalid: {:.2%} ({}/{})'.format(v,accuracy,class_corrects[v],class_n_samples[v],invalid,class_invalids[v],class_n_samples[v]))

    dump_object = {
        'class_corrects':class_corrects,
        'class_invalids':class_invalids,
        'class_total_samples':class_n_samples,
        'confusion_matrix_target':confusion_matrix_target,
        'confusion_matrix_pred':confusion_matrix_pred,
        'confusion_matrix_labels':sorted_labels,
        'global_accuracy':global_accuracy,
        'global_invalids':invalids_perc
    }
    torch.cuda.empty_cache()
    return avg_loss, dump_object

def reload_loaders(clevr_dataset_train, clevr_dataset_test, train_bs, test_bs, state_description = False):
    if not state_description:
        # Use a weighted sampler for training:
        #weights = clevr_dataset_train.answer_weights()
        #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        # Initialize Clevr dataset loaders
        clevr_train_loader = DataLoader(clevr_dataset_train, batch_size=train_bs,
                                        shuffle=False, num_workers=8, collate_fn=utils.collate_samples_from_pixels)
        clevr_test_loader = DataLoader(clevr_dataset_test, batch_size=test_bs,
                                       shuffle=False, num_workers=8, collate_fn=utils.collate_samples_from_pixels)
    else:
        # Initialize Clevr dataset loaders
        clevr_train_loader = DataLoader(clevr_dataset_train, batch_size=train_bs,
                                        shuffle=False, collate_fn=utils.collate_samples_state_description)
        clevr_test_loader = DataLoader(clevr_dataset_test, batch_size=test_bs,
                                       shuffle=False, collate_fn=utils.collate_samples_state_description)
    return clevr_train_loader, clevr_test_loader

def initialize_dataset(clevr_dir, dictionaries, state_description=True, sub_set = 0.0005):
    if not state_description:
        train_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                           transforms.Pad(8),
                                           transforms.RandomCrop((128, 128)),
                                           transforms.RandomRotation(2.8),  # .05 rad
                                           transforms.ToTensor()])
        test_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])
                                          
        clevr_dataset_train = ClevrDataset(clevr_dir, True, dictionaries, train_transforms)
        clevr_dataset_test = ClevrDataset(clevr_dir, False, dictionaries, test_transforms)
        
    else:
        clevr_dataset_train = ClevrDatasetStateDescription(clevr_dir, True, dictionaries)
        clevr_dataset_test = ClevrDatasetStateDescription(clevr_dir, False, dictionaries)

    if sub_set < 1:
        random.seed(10)
        sub_indx = random.sample(range(0, len(clevr_dataset_train)), int(len(clevr_dataset_train)*sub_set))
        clevr_dataset_train = torch.utils.data.Subset(clevr_dataset_train, sub_indx)

        sub_indx = random.sample(range(0, len(clevr_dataset_test)), int(len(clevr_dataset_test)*sub_set))
        clevr_dataset_test = torch.utils.data.Subset(clevr_dataset_test, sub_indx)

    return clevr_dataset_train, clevr_dataset_test 
        
    


def main(args):
    #load hyperparameters from configuration file
    with open(args.config) as config_file: 
        hyp = json.load(config_file)['hyperparams'][args.model]
    #override configuration dropout
    if args.dropout > 0:
        hyp['dropout'] = args.dropout
    if args.question_injection >= 0:
        hyp['question_injection_position'] = args.question_injection

    print('Loaded hyperparameters from configuration {}, model: {}: {}'.format(args.config, args.model, hyp))

    args.model_dirs = args.experiment
    if not os.path.exists(args.model_dirs):
        os.makedirs(args.model_dirs)
    #create a file in this folder containing the overall configuration
    args_str = str(args)
    hyp_str = str(hyp)
    all_configuration = args_str+'\n\n'+hyp_str
    filename = os.path.join(args.model_dirs,'config.txt')
    with open(filename,'w') as config_file:
        config_file.write(all_configuration)

    args.features_dirs = './features'
    args.test_results_dir = args.experiment
    # if not os.path.exists(args.test_results_dir):
        # os.makedirs(args.test_results_dir)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print('Building word dictionaries from all the words in the dataset...')
    dictionaries = utils.build_dictionaries(args.clevr_dir)
    print('Word dictionary completed!')

    print('Initializing CLEVR dataset...')
    clevr_dataset_train, clevr_dataset_test  = initialize_dataset(args.clevr_dir, dictionaries, hyp['state_description'], args.subset)
    print('CLEVR dataset initialized!')

    # Build the model
    args.qdict_size = len(dictionaries[0])
    args.adict_size = len(dictionaries[1])

    model = RN(args, hyp)

    if args.freeze_RN:
        for name, param in model.named_parameters():
            if not name.startswith('rl.m'):
                param.requires_grad = False
                print(f'Freezed {name} layer')

    if torch.cuda.device_count() > 1 and args.cuda:
        model = torch.nn.DataParallel(model)        
        model.module.cuda()  # call cuda() overridden method

    if args.cuda:
        model.cuda()

    start_epoch = 1
    if args.resume:
        filename = args.resume
        if os.path.isfile(filename):
            print('==> loading checkpoint {}'.format(filename))
            checkpoint = torch.load(filename)

            #removes 'module' from dict entries, pytorch bug #3805
            if torch.cuda.device_count() == 1 and any(k.startswith('module.') for k in checkpoint.keys()):
                checkpoint = {k.replace('module.',''): v for k,v in checkpoint.items()}
            if torch.cuda.device_count() > 1 and not any(k.startswith('module.') for k in checkpoint.keys()):
                checkpoint = {'module.'+k: v for k,v in checkpoint.items()}

            model.load_state_dict(checkpoint, strict=False)
            print('==> loaded checkpoint {}'.format(filename))
            start_epoch = int(re.match(r'.*epoch_(\d+).pth', args.resume).groups()[0]) + 1

    if args.conv_transfer_learn:
        if os.path.isfile(args.conv_transfer_learn):
            # TODO: there may be problems caused by pytorch issue #3805 if using DataParallel

            print('==> loading conv layer from {}'.format(args.conv_transfer_learn))
            # pretrained dict is the dictionary containing the already trained conv layer
            pretrained_dict = torch.load(args.conv_transfer_learn)

            if torch.cuda.device_count() == 1:
                conv_dict = model.conv.state_dict()
            else:
                conv_dict = model.module.conv.state_dict()
            
            # filter only the conv layer from the loaded dictionary
            conv_pretrained_dict = {k.replace('conv.','',1): v for k, v in pretrained_dict.items() if 'conv.' in k}

            # overwrite entries in the existing state dict
            conv_dict.update(conv_pretrained_dict)

            # load the new state dict
            if torch.cuda.device_count() == 1:
                model.conv.load_state_dict(conv_dict)
                params = model.conv.parameters()
            else:
                model.module.conv.load_state_dict(conv_dict)
                params = model.module.conv.parameters()

            # freeze the weights for the convolutional layer by disabling gradient evaluation
            # for param in params:
            #     param.requires_grad = False

            print("==> conv layer loaded!")
        else:
            print('Cannot load file {}'.format(args.conv_transfer_learn))

    progress_bar = trange(start_epoch, args.epochs + 1)
    if args.test:
        # perform a single test
        print('Testing epoch {}'.format(start_epoch))
        _, clevr_test_loader = reload_loaders(clevr_dataset_train, clevr_dataset_test, args.batch_size, args.test_batch_size, hyp['state_description'])
        test(clevr_test_loader, model, start_epoch, dictionaries, args)
    else:
        bs = args.batch_size

        # perform a full training
        #TODO: find a better solution for general lr scheduling policies
        candidate_lr = args.lr * args.lr_gamma ** (start_epoch-1 // args.lr_step)
        lr = candidate_lr if candidate_lr <= args.lr_max else args.lr_max
        # lr = 0.005
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
        if args.resume_optimizer: 
            filename = args.resume_optimizer
            checkpoint = torch.load(filename)
            optimizer.load_state_dict(checkpoint)
            print('==> loaded optimizer {}'.format(filename))
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, min_lr=1e-6, verbose=True)
        scheduler = lr_scheduler.StepLR(optimizer, args.lr_step, gamma=args.lr_gamma)
        scheduler.last_epoch = start_epoch
        best_loss = float('inf')
        best_epoch = float('inf')
        print('Training ({} epochs) is starting...'.format(args.epochs))
        for epoch in progress_bar:
            
            if(((args.bs_max > 0 and bs < args.bs_max) or args.bs_max < 0 ) and (epoch % args.bs_step == 0 or epoch == start_epoch)):
                bs = math.floor(args.batch_size * (args.bs_gamma ** (epoch // args.bs_step)))
                if bs > args.bs_max and args.bs_max > 0:
                    bs = args.bs_max
                clevr_train_loader, clevr_test_loader = reload_loaders(clevr_dataset_train, clevr_dataset_test, bs, args.test_batch_size, hyp['state_description'])

                #restart optimizer in order to restart learning rate scheduler
                #for param_group in optimizer.param_groups:
                #    param_group['lr'] = args.lr
                #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, step, min_lr)
                print('Dataset reinitialized with batch size {}'.format(bs))
            
            if((args.lr_max > 0 and scheduler.get_lr()[0]<args.lr_max) or args.lr_max < 0):
                scheduler.step()
                    
            print('Current learning rate: {}'.format(optimizer.param_groups[0]['lr']))
                
            if args.comet:           
                with experiment.train():
                    # TRAIN
                    progress_bar.set_description('TRAIN')
                    accuracy = train(clevr_train_loader, model, optimizer, epoch, args)
                    experiment.log_metrics({
                        'accuracy' : accuracy,
                        'epoch' : epoch,
                    })

                with experiment.test():
                    # TEST
                    progress_bar.set_description('TEST')
                    test_loss, results = test(clevr_test_loader, model, epoch, dictionaries, args)
                    for v in results['class_total_samples'].keys():
                        accuracy = 0
                        invalid = 0
                        if results['class_total_samples'][v] != 0:
                            accuracy = results['class_corrects'][v] / results['class_total_samples'][v]
                            invalid = results['class_invalids'][v] / results['class_total_samples'][v]
                        experiment.log_metrics({
                            f'{v}_accuracy' : accuracy,
                            f'{v}_invalid' : invalid})
                    experiment.log_metrics({
                        'accuracy' : results['global_accuracy'],
                        'invalids' : results['global_invalids'],
                        'loss' : test_loss,
                        'epoch' : epoch,
                    })

                if test_loss < best_loss:
                    print('Saving weights for epoch {}'.format(epoch))
                    # SAVE MODEL AND OPTIMIZER
                    weights_filename = os.path.join(args.model_dirs, 'best_weights.pth')
                    torch.save(model.state_dict(), weights_filename)
                    optimizer_filename = os.path.join(args.model_dirs, 'best_optimizer.pth')
                    torch.save(optimizer.state_dict(), optimizer_filename)
                    # dump results on file
                    results_filename = os.path.join(args.model_dirs, f'test_epoch_{epoch}.pickle')
                    pickle.dump(results, open(results_filename,'wb'))
                    best_loss = test_loss
                    best_epoch = epoch

            else:
                # TRAIN
                progress_bar.set_description('TRAIN')
                accuracy = train(clevr_train_loader, model, optimizer, epoch, args)
                # TEST
                progress_bar.set_description('TEST')
                test_loss, results = test(clevr_test_loader, model, epoch, dictionaries, args)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Relational-Network CLEVR')
    parser.add_argument('--batch-size', type=int, default=640, metavar='N',
                        help='input batch size for training (default: 640)')
    parser.add_argument('--test-batch-size', type=int, default=640,
                        help='input batch size for training (default: 640)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=0.000005, metavar='LR',
                        help='learning rate (default: 0.000005)')
    parser.add_argument('--clip-norm', type=int, default=50,
                        help='max norm for gradients; set to 0 to disable gradient clipping (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str,
                        help='resume from model stored')
    parser.add_argument('--resume_optimizer', type=str,
                        help='resume from optimizer stored')
    parser.add_argument('--freeze_RN', type=bool, default=False,
                        help='freeze RN weights')
    parser.add_argument('--clevr-dir', type=str, default='.',
                        help='base directory of CLEVR dataset')
    parser.add_argument('--model', type=str, default='original-fp',
                        help='which model is used to train the network')
    parser.add_argument('--experiment', type=str, default='pruebas',
                        help='experiment name')
    parser.add_argument('--no-invert-questions', action='store_true', default=False,
                        help='invert the question word indexes for LSTM processing')
    parser.add_argument('--test', action='store_true', default=False,
                        help='perform only a single test. To use with --resume')
    parser.add_argument('--conv-transfer-learn', type=str,
                    help='use convolutional layer from another training')
    parser.add_argument('--lr-max', type=float, default=0.0005,
                        help='max learning rate')
    parser.add_argument('--lr-gamma', type=float, default=2, 
                        help='increasing rate for the learning rate. 1 to keep LR constant.')
    parser.add_argument('--lr-step', type=int, default=20,
                        help='number of epochs before lr update')
    parser.add_argument('--bs-max', type=int, default=-1,
                        help='max batch-size')
    parser.add_argument('--bs-gamma', type=float, default=1, 
                        help='increasing rate for the batch size. 1 to keep batch-size constant.')
    parser.add_argument('--bs-step', type=int, default=20, 
                        help='number of epochs before batch-size update')
    parser.add_argument('--dropout', type=float, default=-1,
                        help='dropout rate. -1 to use value from configuration')
    parser.add_argument('--config', type=str, default='config.json',
                        help='configuration file for hyperparameters loading')
    parser.add_argument('--question-injection', type=int, default=-1, 
                        help='At which stage of g function the question should be inserted (0 to insert at the beginning, as specified in DeepMind model, -1 to use configuration value)')
    parser.add_argument('--subset', type=float, default=1.0,
                        help='percentage of the dataset')
    parser.add_argument('--l1-lambd', type=float, default=1.0,
                        help='L1 lambd for loss')
    parser.add_argument('--comet', type=int, default=1,
                        help='Log to comet')
    parser.add_argument('--resume_comet', type=str, default='',
                        help='Log to comet')
    args = parser.parse_args()
    args.invert_questions = not args.no_invert_questions
    if args.comet:
        experiment = Experiment(api_key="VD0MYyhx0BQcWhxWvLbcalX51",
                        project_name="rn", workspace="adaptive-weights")
        experiment.set_name(args.experiment)
        experiment.log_parameters({
            'batch_size' : args.batch_size,
            'test_batch_size' : args.test_batch_size,
            'subset' : args.subset,
            'l1-lambd' : args.l1_lambd
            })
    if args.resume_comet:
        print(f'Resumed comet with key {args.resume_comet}')
        experiment = ExistingExperiment(api_key="VD0MYyhx0BQcWhxWvLbcalX51", 
                        previous_experiment=args.resume_comet)
        experiment.log_parameters({
            'batch_size' : args.batch_size,
            'test_batch_size' : args.test_batch_size,
            'subset' : args.subset,
            'l1-lambd' : args.l1_lambd
            })
    main(args)
