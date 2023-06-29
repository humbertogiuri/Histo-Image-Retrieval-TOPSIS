from db import Database
from argparse import ArgumentParser, ArgumentTypeError
import models as models
from PIL import Image
import time
import torch
import os

class ImageRetriever:
    def __init__(self, db_name, model):
        self.db = Database(db_name, model, True)

    def retrieve(self, image, nrt_neigh=10, topsis=False):
        return self.db.search(image, nrt_neigh)[0] if not topsis else self.db.search(image, 198)
    
    def get_clinical_data(self, name_image, path_csv_train="./CLINICAL/ndbufes_TaskIV_parsed_folders.csv", path_csv_test="./CLINICAL/ndbufes_TaskIV_parsed_test.csv"):
        import pandas as pd

        df_train = pd.read_csv(path_csv_train).drop(columns=['larger_size', 'TaskIV', 'folder'])
        df_test = pd.read_csv(path_csv_test).drop(columns=['larger_size', 'TaskIV', 'folder'])

        query_series = df_test[df_test['path'] == name_image].squeeze()

        clinical_data_train = [(row['path'], row.drop('path').tolist()) for _, row in df_train.iterrows()]
        clinical_data_query = (query_series['path'], query_series.drop(labels=['path']).to_list())

        return clinical_data_query, clinical_data_train
    
    def get_hamming_distances(self, clinical_data_query, clinical_data_train):
        from scipy.spatial.distance import hamming

        name_query, clinical_vector_query = clinical_data_query[0], clinical_data_query[1]
        
        return [(name, hamming(clinical_vector_query, clinical_vector)) for name, clinical_vector in clinical_data_train]
    
    def apply_topsis(self, hamming_distances, l2_distances):
        from topsis import Topsis
        import numpy as np

        def get_tupla(hamming_distances, name):
            for x in hamming_distances:
                if x[0] == name.split('/')[-1]:
                    return x
       
        topsis_data = []
        
        for idx, name in enumerate(l2_distances[0]):
            hamming_tupla = get_tupla(hamming_distances, name)
            topsis_data.append((name, l2_distances[1][idx], hamming_tupla[1]))

        topsis_distances = np.vstack(topsis_data)[:, 1:]

        weights = [0.5, 0.5]
        criterias = [False, False]

        topsis = Topsis(topsis_distances, weights, criterias)
        topsis.calc()

        best_alternatives = topsis.rank_to_best_similarity()
        #print(best_alternatives)
        #print([topsis_data[i - 1][0] for i in best_alternatives])
        return [topsis_data[i - 1][0] for i in best_alternatives]

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--path',
        help='path to the image',
    )

    parser.add_argument(
        '--extractor',
        help='feature extractor that is used',
        default='densenet'
    )

    parser.add_argument(
        '--db_name',
        help='name of the database',
        default='db'
    )

    parser.add_argument(
        '--num_features',
        help='number of features to extract',
        default=128,
        type=int
    )

    parser.add_argument(
        '--weights',
        help='file that contains the weights of the network',
        default='weights'
    )

    parser.add_argument(
        '--dr_model',
        action="store_true"
    )

    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int
    )

    parser.add_argument(
        '--nrt_neigh',
        default=5,
        type=int
    )

    parser.add_argument(
        '--topsis',
        default=False,
        type=bool
    )

    args = parser.parse_args()

    if args.gpu_id >= 0:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    if args.path is None:
        print(usage)
        exit(-1)

    if not os.path.isfile(args.path):
        print('Path mentionned is not a file')
        exit(-1)

    model = models.Model(model=args.extractor, num_features=args.num_features, name=args.weights,
                           use_dr=args.dr_model, device=device)

    retriever = ImageRetriever(args.db_name, model)

    images_retrieved = retriever.retrieve(Image.open(args.path).convert('RGB'), args.nrt_neigh, args.topsis)
    names, distances = images_retrieved[0], images_retrieved[1][0]
    
    clinical_data_query, clinical_data = retriever.get_clinical_data(name_image=args.path.split('/')[-1])

    hamming_distances = retriever.get_hamming_distances(clinical_data_query, clinical_data)

    alternatives = retriever.apply_topsis(hamming_distances, (names, distances))[:args.nrt_neigh]
    print(alternatives)
    for n in alternatives:
        Image.open(n).show()
    