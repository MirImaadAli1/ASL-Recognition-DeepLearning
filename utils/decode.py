import torch
import pyctcdecode  # Import pyctcdecode for CTC decoding
import numpy as np
from itertools import groupby
import torch.nn.functional as F

class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        
        # Create vocabulary from gloss_dict
        vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
        
        # Initialize the CTC decoder without KenLM for decoding
        self.ctc_decoder = None

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        
        return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        '''
        Performs beam search decoding without using a KenLM model.
        '''
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        
        vid_lgt = vid_lgt.cpu()
        print(f"vid_lgt values: {vid_lgt.tolist()}")
        
        # Perform greedy decoding if no KenLM model is provided
        if self.ctc_decoder is None:
            ret_list = []
            for batch_idx in range(len(nn_output)):
                # Decode the most probable sequence using argmax
                decoded_indices = nn_output[batch_idx].argmax(dim=-1)
                
                # Remove the blank indices
                decoded_indices = decoded_indices[decoded_indices != self.blank_id]
                
                # Map indices to glosses
                decoded_gloss = [self.i2g_dict[int(gloss_id)] for gloss_id in decoded_indices]
                
                # Remove repeating glosses
                decoded_gloss = [gloss for gloss, _ in groupby(decoded_gloss)]
                
                print("decoded shii ", decoded_gloss)

                # Only add to ret_list if decoded_gloss is non-empty
                if decoded_gloss:  # This ensures empty lists are not added
                    ret_list.append(decoded_gloss)
            
            # Print the filtered ret_list
            print("Filtered ret_list:", ret_list)

        return ret_list



