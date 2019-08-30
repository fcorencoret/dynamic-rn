import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from MultiheadAttention import MultiheadAttention
import pdb

MULTIHEADATTENTION_HEADS = 1


class MetricActivation(object):
    def __init__(self, threshold_mask = 0.3):
        super(MetricActivation, self).__init__()

        self.threshold_mask = threshold_mask

    def getDiff(self, x, mask):
        m = (mask > self.threshold_mask).float().sum()
        a = (x > 0).float().sum()
        b = ((mask > self.threshold_mask).float()*(x > 0).float()).sum()

        # a - b = Las neuronas que apagamos
        # m - b = Venian apagadas y las mascara no las vio

        return b / x.nelement(), (a - b) / x.nelement(), (m - b) / x.nelement()

    def changeMask(self, mask, t = 'ones'):
        if t == 'ones':
            return torch.ones_like(mask)
        else:
            return mask[:,torch.randperm(mask.size(-1))]

class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)
        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = self.batchNorm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.batchNorm4(x)
        x = F.relu(x)
        return x

class QuestionEmbedModel(nn.Module):
    def __init__(self, in_size, embed=32, hidden=128):
        super(QuestionEmbedModel, self).__init__()
        
        self.wembedding = nn.Embedding(in_size + 1, embed)  #word embeddings have size 32
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)  # Input dim is 32, output dim is the question embedding
        self.hidden = hidden
        
    def forward(self, question):
        #calculate question embeddings
        wembed = self.wembedding(question)
        # wembed = wembed.permute(1,0,2) # in lstm minibatches are in the 2-nd dimension
        self.lstm.flatten_parameters()
        _, hidden = self.lstm(wembed) # initial state is set to zeros by default
        qst_emb = hidden[0] # hidden state of the lstm. qst = (B x 128)
        #qst_emb = qst_emb.permute(1,0,2).contiguous()
        #qst_emb = qst_emb.view(-1, self.hidden*2)
        qst_emb = qst_emb[0]
        
        return qst_emb

class RelationalLayerBase(nn.Module):
    def __init__(self, in_size, out_size, qst_size, hyp):
        super().__init__()

        # f_fc1
        self.f_fc1 = nn.Linear(hyp["g_layers"][-1], hyp["f_fc1"])
        self.mha_fc1 = MultiheadAttention(hyp["g_layers"][-1], MULTIHEADATTENTION_HEADS)
        self.identity_fc1 = nn.Identity()
        # f_fc2
        self.f_fc2 = nn.Linear(hyp["f_fc1"], hyp["f_fc2"])
        self.mha_fc2 = MultiheadAttention(hyp["f_fc1"], MULTIHEADATTENTION_HEADS)
        self.identity_fc2 = nn.Identity()
        # f_fc3
        self.f_fc3 = nn.Linear(hyp["f_fc2"], out_size)
        self.mha_fc3 = MultiheadAttention(hyp["f_fc2"], MULTIHEADATTENTION_HEADS)
        self.identity_fc3 = nn.Identity()
    
        self.dropout = nn.Dropout(p=hyp["dropout"])
        
        self.on_gpu = False
        self.hyp = hyp
        self.qst_size = qst_size
        self.in_size = in_size
        self.out_size = out_size

    def cuda(self, device=None):
        self.on_gpu = True
        super().cuda(device)
    
class RelationalLayer(RelationalLayerBase):
    def __init__(self, in_size, out_size, qst_size, hyp, extraction=False):
        super().__init__(in_size, out_size, qst_size, hyp)

        self.quest_inject_position = hyp["question_injection_position"]
        self.in_size = in_size

        self.wa = MetricActivation()

	    #create all g layers
        self.g_layers = []
        self.g_layers_size = hyp["g_layers"]

        #create all multiheadattention layers
        self.mha_layers = []
        self.identity_layers = []

        for idx,g_layer_size in enumerate(hyp["g_layers"]):
            in_s = in_size if idx==0 else hyp["g_layers"][idx-1]
            out_s = g_layer_size
            if idx==self.quest_inject_position:
                #create the h layer. Now, for better code organization, it is part of the g layers pool. 
                l = nn.Linear(in_s+qst_size, out_s)
                mha = MultiheadAttention(in_s+qst_size, MULTIHEADATTENTION_HEADS)
            else:
                #create a standard g layer.
                l = nn.Linear(in_s, out_s)
                mha = MultiheadAttention(in_s, MULTIHEADATTENTION_HEADS)
            self.g_layers.append(l)
            self.mha_layers.append(mha)
            self.identity_layers.append(nn.Identity())


        self.g_layers = nn.ModuleList(self.g_layers)
        self.mha_layers = nn.ModuleList(self.mha_layers)
        self.identity_layers = nn.ModuleList(self.identity_layers)
        self.extraction = extraction
    
    def forward(self, x, qst):
        # x = (B x 8*8 x 24)
        # qst = (B x 128)
        """g"""
        b, d, k = x.size()
        qst_size = qst.size()[1]
        l1_reg = 0
        
        # add question everywhere
        qst = torch.unsqueeze(qst, 1)                      # (B x 1 x 128)
        query = qst.clone().transpose(1, 0)
        qst = qst.repeat(1, d, 1)                       # (B x 64 x 128)
        qst = torch.unsqueeze(qst, 2)                      # (B x 64 x 1 x 128)
        
        # cast all pairs against each other
        x_i = torch.unsqueeze(x, 1)                   # (B x 1 x 64 x 26)
        x_i = x_i.repeat(1, d, 1, 1)                    # (B x 64 x 64 x 26)
        x_j = torch.unsqueeze(x, 2)                   # (B x 64 x 1 x 26)
        #x_j = torch.cat([x_j, qst], 3)
        x_j = x_j.repeat(1, 1, d, 1)                    # (B x 64 x 64 x 26)
        
        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3)                  # (B x 64 x 64 x 2*26)
        
        # reshape for passing through network
        x_ = x_full.view(b * d**2, self.in_size)

        results_wa = {}

        #create g and inject the question at the position pointed by quest_inject_position.
        for idx, (g_layer, mha_layer, g_layer_size, identity) in enumerate(zip(self.g_layers, self.mha_layers, self.g_layers_size, self.identity_layers)):
            if idx==self.quest_inject_position:
                in_size = self.in_size if idx==0 else self.g_layers_size[idx-1]

                # questions inserted
                x_img = x_.view(b,d,d,in_size)
                qst = qst.repeat(1,1,d,1)
                x_concat = torch.cat([x_img,qst],3) #(B x 64 x 64 x 128 + 2 * 26)

                # h layer
                x_ = x_concat.view(b*(d**2),in_size+self.qst_size)
                x_ = g_layer(x_)
                x_ = F.relu(x_)
            else:
                x_ = g_layer(x_)
                x_ = F.relu(x_)
                # Pass through multiheadattention layer
                weights = torch.unsqueeze(g_layer.weight, 0).repeat(b, 1, 1).transpose(1, 0)
                _, attn_output_weights = mha_layer(query, weights, weights)
                l1_reg += (attn_output_weights.abs().sum() / (attn_output_weights.size(0) * attn_output_weights.size(2)))
                attn_output_weights = attn_output_weights.repeat(1, d**2, 1)
                # Apply attn_output_weights to x_

                #print(self.wa.getDiff(x_.view(b, d**2, g_layer_size), attn_output_weights))
                results_wa['fc_'+str(idx)+'_both'], results_wa['fc_'+str(idx)+'_turn_off'], results_wa['fc_'+str(idx)+'_already_turn_off'] = self.wa.getDiff(x_.view(b, d**2, g_layer_size), attn_output_weights)

                x_ = x_.view(b, d**2, g_layer_size) * attn_output_weights
                x_ = x_.view(b * (d ** 2), g_layer_size)
            x_ = identity(x_)

        if self.extraction:
            return None
        
        # reshape again and sum
        x_g = x_.view(b, d**2, self.g_layers_size[-1])
        x_g = x_g.sum(1).squeeze(1)
        
        """f"""
        # f_fc1
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        weights = torch.unsqueeze(self.f_fc1.weight, 0).repeat(b, 1, 1).transpose(1, 0)
        _, attn_output_weights = self.mha_fc1(query, weights, weights)
        l1_reg += (attn_output_weights.abs().sum() / (attn_output_weights.size(0) * attn_output_weights.size(2)))

        #print(self.wa.getDiff(x_f, attn_output_weights.squeeze(1)))
        results_wa['fc_1_both'], results_wa['fc_1_turn_off'], results_wa['fc_1_already_turn_off'] = self.wa.getDiff(x_f, attn_output_weights.squeeze(1))


        x_f = x_f * attn_output_weights.squeeze(1)
        x_f = self.identity_fc1(x_f)
        # f_fc2
        x_f = self.f_fc2(x_f)
        x_f = self.dropout(x_f)
        x_f = F.relu(x_f)
        weights = torch.unsqueeze(self.f_fc2.weight, 0).repeat(b, 1, 1).transpose(1, 0)
        _, attn_output_weights = self.mha_fc2(query, weights, weights)
        l1_reg += (attn_output_weights.abs().sum() / (attn_output_weights.size(0) * attn_output_weights.size(2)))

        #print(self.wa.getDiff(x_f, attn_output_weights.squeeze(1)))
        results_wa['fc_2_both'], results_wa['fc_2_turn_off'], results_wa['fc_2_already_turn_off'] = self.wa.getDiff(x_f, attn_output_weights.squeeze(1))

        x_f = x_f * attn_output_weights.squeeze(1)
        x_f = self.identity_fc2(x_f)
        # f_fc3
        x_f = self.f_fc3(x_f)
        weights = torch.unsqueeze(self.f_fc3.weight, 0).repeat(b, 1, 1).transpose(1, 0)
        _, attn_output_weights = self.mha_fc3(query, weights, weights)
        l1_reg += (attn_output_weights.abs().sum() / (attn_output_weights.size(0) * attn_output_weights.size(2)))

        #print(self.wa.getDiff(x_f, attn_output_weights.squeeze(1)))
        results_wa['fc_3_both'], results_wa['fc_3_turn_off'], results_wa['fc_3_already_turn_off'] = self.wa.getDiff(x_f, attn_output_weights.squeeze(1))

        x_f = x_f * attn_output_weights.squeeze(1)
        x_f = self.identity_fc3(x_f)
        return F.log_softmax(x_f, dim=1), l1_reg, results_wa

class RN(nn.Module):
    def __init__(self, args, hyp, extraction=False):
        super(RN, self).__init__()
        self.coord_tensor = None
        self.on_gpu = False
        
        # CNN
        self.conv = ConvInputModel()
        self.state_desc = hyp['state_description']            
            
        # LSTM
        hidden_size = hyp["lstm_hidden"]
        self.text = QuestionEmbedModel(args.qdict_size, embed=hyp["lstm_word_emb"], hidden=hidden_size)
        
        # RELATIONAL LAYER
        self.rl_in_size = hyp["rl_in_size"]
        self.rl_out_size = args.adict_size
        self.rl = RelationalLayer(self.rl_in_size, self.rl_out_size, hidden_size, hyp, extraction) 
        if hyp["question_injection_position"] != 0:          
            print('Supposing IR model')
        else:     
            print('Supposing original DeepMind model')

    def forward(self, img, qst_idxs):
        if self.state_desc:
            x = img # (B x 12 x 8)
        else:
            x = self.conv(img)  # (B x 24 x 8 x 8)
            b, k, d, _ = x.size()
            x = x.view(b,k,d*d) # (B x 24 x 8*8)
            
            # add coordinates
            if self.coord_tensor is None or torch.cuda.device_count() == 1:
                self.build_coord_tensor(b, d)                  # (B x 2 x 8 x 8)
                self.coord_tensor = self.coord_tensor.view(b,2,d*d) # (B x 2 x 8*8)
            
            x = torch.cat([x, self.coord_tensor], 1)    # (B x 24+2 x 8*8)
            x = x.permute(0, 2, 1)    # (B x 64 x 24+2)
        
        qst = self.text(qst_idxs)
        y = self.rl(x, qst)
        return y
       
    # prepare coord tensor
    def build_coord_tensor(self, b, d):
        coords = torch.linspace(-d/2., d/2., d)
        x = coords.unsqueeze(0).repeat(d, 1)
        y = coords.unsqueeze(1).repeat(1, d)
        ct = torch.stack((x,y))
        # broadcast to all batches
        # TODO: upgrade pytorch and use broadcasting
        ct = ct.unsqueeze(0).repeat(b, 1, 1, 1)
        self.coord_tensor = Variable(ct, requires_grad=False)
        if self.on_gpu:
            self.coord_tensor = self.coord_tensor.cuda()
    
    def cuda(self, device=None):
        self.on_gpu = True
        self.rl.cuda(device)
        super(RN, self).cuda(device)
        
if __name__ == '__main__':
    # Training settings
    import argparse
    import json

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
                        help='At which stage of g function the question should be inserted (0 to insert at the beginning, as specified in DeepMind model, -1 to use configuration value), -2 to delete question from G layers')
    parser.add_argument('--subset', type=float, default=1.0,
                        help='percentage of the dataset')
    parser.add_argument('--l1-lambd', type=float, default=1.0,
                        help='L1 lambd for loss')
    parser.add_argument('--comet', type=int, default=1,
                        help='Log to comet')
    parser.add_argument('--resume_comet', type=str, default='',
                        help='Log to comet')
    parser.add_argument('--dataset', type=str, default='clevr')


    args = parser.parse_args()
    args.invert_questions = not args.no_invert_questions
    args.qdict_size = 1000
    args.adict_size = 400

    with open(args.config) as config_file: 
        hyp = json.load(config_file)['hyperparams'][args.model]

    img = torch.FloatTensor(16, 3, 128, 128)
    qst = torch.LongTensor(16, 32).random_(0, 10)


    model = RN(args, hyp)

    for i in range(10):
        model(img, qst)