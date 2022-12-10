import os 
import torch
import numpy as np


def load_flamingo_weights(cur_model, path):
    """
    Load Flamingo weights after adding additional token to the vocabulary.
    directly loading gives error.
    """
    print("Flamingo Weights are loading...")
    model_dict = cur_model.state_dict()
    print("Total num elements in model's state_dict:", len(model_dict))

    if os.getcwd().startswith('/Users/caghankoksal'):
            state_dict = torch.load(path, map_location='cpu')['state_dict']  # pretrained weights
    else:
            state_dict = torch.load(path)['state_dict']  # pretrained weights
    print("Total num elements in pretrained weights:", len(state_dict))

    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and \
            not k.startswith('flamingo_palm.token_emb.weight') and\
            not k.startswith('flamingo_palm.to_logits.1.weight')  \
            }
    
    print("State dict is loaded with {} keys".format(len(pretrained_dict)))
    cur_model.load_state_dict(pretrained_dict, strict=False)
    cur_model.state_dict()["flamingo_palm.token_emb.weight"][:-1, :] = state_dict["flamingo_palm.token_emb.weight"]
    cur_model.state_dict()["flamingo_palm.to_logits.1.weight"][:-1, :] = state_dict["flamingo_palm.to_logits.1.weight"]


def print_hyperparams(hparams):
    for k,v in hparams.items():
        print(k,v)


import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue


class BeamSearchNode(object):
    def __init__(self, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

def beam_decode(batch,cur_model, tokenizer, top_k = 3, beam_width = 3, max_len=50):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    beam_width = beam_width
    topk = top_k  # how many sentence do you want to generate
    decoded_batch = []

    batch_size = 1



    # decoding goes sentence by sentence
    for idx in range(batch_size):
        

        # Start with the start of the sentence token
        decoder_input   = torch.tensor([tokenizer.encode("<|endoftext|> <image> question: "+batch["question"][0] + ' <EOQ>'+ ' answer:')]) 

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))
        #print("Number of required : ",number_required)

        # starting node -   previous node, word id, logp, length
        node = BeamSearchNode(None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 200: break

            # fetch the best node
            score, n = nodes.get()
            #print("Best node is chosen with score : ",score)
            decoder_input = n.wordid
            #print("Best node is chosen with wordid : ",decoder_input)
            #decoder_hidden = n.h

            #print("n.wordid[:,-1]  : ",n.wordid[:,-1])
            if n.wordid[:,-1].item() == 50259 and n.prevNode is not None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue
            if len(n.wordid) > max_len:
                endnodes.append((score, n))
                if len(endnodes) >= number_required:
                    break
                else:
                    continue
            #out = cur_model({'image': batch['image'],'input_ids': decoder_input })
            out,_ = cur_model({'image': batch['image'],'input_ids': decoder_input, "index_eoq": batch["index_eoq"],
            "targets": batch["targets"],"label": batch["label"]})
            #print("Out shape : ",out.shape)
            logits = out[:, -1, :]
            indices_to_remove = logits < torch.topk(logits, 10)[0][..., -1, None]
            logits[indices_to_remove] = np.NINF
            logits = F.log_softmax(logits, dim=-1)
            #print("Logits shape : ",logits.shape)
            
            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(logits, beam_width)
            nextnodes = []
            #print("Log Prob : ",log_prob.shape)
            #print("Indexes : ",indexes.shape)

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()
                cur_context = torch.cat([decoder_input, decoded_t], dim=-1)
                
                node = BeamSearchNode(n, cur_context, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                #print("Current score ",score)
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            #print("Score : ",score, n.wordid)
            utterance = []
            utterance.append(n.wordid)
            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch


import yaml

# Function to load yaml configuration file
def load_config(CONFIG_PATH, config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


