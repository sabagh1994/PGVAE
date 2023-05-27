# Most of the scripts are adapted from ECNet (Luo et al, Nature Comm):

import numpy as np
import pandas as pd
import collections


'''
Amino acide encoding modified from 
https://github.com/openvax/mhcflurry/blob/74b751e6d72605eef4a49641d364066193541b5a/mhcflurry/amino_acid.py
'''
COMMON_AMINO_ACIDS_INDEX = collections.OrderedDict(    
    {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 
     'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 
     'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 
     'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, '-': 20})
AMINO_ACIDS = list(COMMON_AMINO_ACIDS_INDEX.keys())

AMINO_ACID_INDEX = collections.OrderedDict(
    {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 
     'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 
     'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 
     'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,      
     'X': 20, 'Z': 20, 'B': 20, 'J': 20, '-': 20})

ENCODING_DATA_FRAMES = {
    "one-hot": pd.DataFrame([
        [1 if i == j else 0 for i in range(len(AMINO_ACIDS))]
        for j in range(len(AMINO_ACIDS))
    ], index=AMINO_ACIDS, columns=AMINO_ACIDS)
}

def convert_idx_array_to_aas_ecnet(gen_seqs_all):
    aa_letter_indOrdered = np.array(ENCODING_DATA_FRAMES["one-hot"].columns)
    # convert one-hot encoded seqs to aa/string sequences
    gen_seqs_aa = aa_letter_indOrdered[gen_seqs_all]
    assert gen_seqs_aa.shape == gen_seqs_all.shape
    assem_seqs = ["".join(seq) for seq in gen_seqs_aa] # aa strings
    
    return assem_seqs

def index_encoding(sequences, letter_to_index_dict):
    '''
    Modified from https://github.com/openvax/mhcflurry/blob/master/mhcflurry/amino_acid.py#L110-L130
    
    Parameters
    ----------
    sequences: list of equal-length sequences
    letter_to_index_dict: char -> int

    Returns
    -------
    np.array with shape (#sequences, length of sequences)
    '''

    df = pd.DataFrame(iter(s) for s in sequences)
    encoding = df.replace(letter_to_index_dict)
    encoding = encoding.values.astype(np.int)
    return encoding

def vector_encoding(sequences, letter_to_vector_df):
    '''
    Modified from https://github.com/openvax/mhcflurry/blob/master/mhcflurry/amino_acid.py#L133-L158
    
    Parameters
    ----------
    sequences: list of equal-length sequences
    letter_to_vector_df: char -> vector embedding

    Returns
    -------
    np.array with shape (#sequences, length of sequences, embedding dim)
    '''
    index_encoded_sequences = index_encoding(sequences, AMINO_ACID_INDEX)
    (num_sequences, sequence_length) = index_encoded_sequences.shape
    target_shape = (num_sequences, sequence_length, letter_to_vector_df.shape[0])
    result = letter_to_vector_df.iloc[index_encoded_sequences.flatten()].values.reshape(target_shape)
    return result

def encode_sequence(sequences, method, dataset_name=None, lm_bsz=64):    
    '''
    dataset_name is used for dataset dependent encoding method,
    e.g., CCMPred profile encoding.
    '''
    if method == 'index':
        encoding = index_encoding(sequences, AMINO_ACID_INDEX)
    elif method == 'BLOSUM62':
        encoding = vector_encoding(sequences, ENCODING_DATA_FRAMES['BLOSUM62'])
    elif method == 'one-hot':
        encoding = vector_encoding(sequences, ENCODING_DATA_FRAMES['one-hot'])
    elif method.startswith('doc2vec'):
        doc2vec_model_name = method.split('-')[1]
        encoding = doc2vec_embedding(sequences, doc2vec_model_name)
    elif method.startswith('tape'):
        tape_model = 'lstm' if method == 'tape' else method.split('-')[1]
        encoding = tape_embedding(sequences, tape_model=tape_model, lm_bsz=lm_bsz)
    else:
        raise NotImplementedError('{} not implemented'.format(method))
    return encoding


def convert_idx_to_aas(seqs_index_arr):
    """
        convert index encoded seqs to amino acid seqs
        seqs_index_arr: (torch.tensor) 
        seqs_aa: (array object)
    """
    aa_letter_indOrdered = np.array(ENCODING_DATA_FRAMES["one-hot"].columns)
    # convert index-encoded seqs to aa/string sequences
    seqs_aa = aa_letter_indOrdered[seqs_index_arr]
    assert seqs_aa.shape == seqs_index_arr.shape
    return seqs_aa