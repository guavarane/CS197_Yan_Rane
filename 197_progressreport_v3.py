import os
import pickle
import rdkit
import selfies
from sklearn.preprocessing import OneHotEncoder
import csv
import numpy as np
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


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
        esm_emb_prot = esm_embeddings_all[i]

        mol = Chem.MolFromSmiles(smile_string)

        bit_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        bit_vec = np.array(bit_vec)
        bit_vec = list(bit_vec)
        esm_emb_prot = np.array(esm_emb_prot)
        esm_emb_prot = list(esm_emb_prot)
        bit_vec.extend(esm_emb_prot)
        bit_vec.append(binding_mes)
        master_data.append(bit_vec)
        print(len(bit_vec), len(esm_emb_prot))
        '''
            Diff Method
        '''

            # Below numbers were calculated by finding the max length/width of the largest one hot encoding
            # We will balance our data so the one hot encoding is a flattened 1D vector of length 53514
            # max_length = 1982
            # max_width = 27
            # max_len = 53514
            # encoded_row = encoder.fit_transform([[char] for char in smile_string])
            # cur_width = len(encoded_row[0])
            # one_hot_encoding_line = []
            # for row in encoded_row:
            #     for elem in row:
            #         one_hot_encoding_line.append(elem)
            #     for j in range(cur_width, max_width):
            #         # fill empty zeros to match the dimensions of the largest one hot encoding
            #         # currently, each character has a 27 length encoding, so if our current encoding is less than that
            #         # just fill it with a bunch of 0's
            #         one_hot_encoding_line.append(0)
            # for k in range(len(one_hot_encoding_line), max_len):
            #     # just add 0's for the remaining set of lines
            #     one_hot_encoding_line.append(0)
            #
            # np_arr = np.array(one_hot_encoding_line)
            # tensor_array = esm_emb_prot.numpy()
            # final_line = np.concatenate((np_arr, tensor_array))
            # final_line = list(final_line)
            # res = [final_line, binding_mes]
            # master_data.append(res)
            # print(i)
            # if i == 0:
            #     print(res)
            # writer.writerow(res)

    with open('/Users/gauravrane/Documents/Spring_2023/CS197/CS197_dataset/Master_bit.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        #
        # for elem in master_data:
        #     writer.writerow(elem)
        # Write the data to the CSV file
        writer.writerows(master_data)

    print(f"Data has been written successfully.")


if __name__ == '__main__':
    basic_commands()
