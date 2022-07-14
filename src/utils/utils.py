import os 
import torch


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