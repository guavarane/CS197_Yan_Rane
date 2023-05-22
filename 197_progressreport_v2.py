import os
import pickle
import rdkit
import selfies
from sklearn.preprocessing import OneHotEncoder

def basic_commands():

    # Data Loading Method
    with open('/Users/gauravrane/Documents/Spring_2023/CS197/CS197_dataset/DATASET.csv') as f:
        dataset_lines = f.readlines()

    with open('/Users/gauravrane/Documents/Spring_2023/CS197/CS197_dataset/ESM_embedding.pickle', 'rb') as input_file:
        esm_embeddings_all = pickle.load(input_file)

    encoder = OneHotEncoder(sparse_output=False)
    master_data = []
    for i, line in enumerate(dataset_lines):
        A = line.split(',')
        smile_string = A[0]
        k_d = A[2].strip()
        if k_d[0] == '>' or k_d[0] == '<':
            binding_mes = float(k_d[1:])
        else:
            binding_mes = float(k_d)
        esm_emb_prot = esm_embeddings_all[0]

        encoded_row = encoder.fit_transform([[char] for char in smile_string])
        res = (encoded_row, esm_emb_prot, binding_mes)
        master_data.append(res)

if __name__ == '__main__':
    basic_commands()

