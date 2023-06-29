from models import Model
import torch
import db as db
from PIL import Image
from argparse import ArgumentParser, ArgumentTypeError
from collections import Counter, defaultdict
from torch.utils.data import Dataset
import os
import numpy as np
import sklearn
import time

class TestDataset(Dataset):
    def __init__(self, root, measure, generalise):
        self.root = root

        self.dic_img = defaultdict(list)
        self.img_list = []

        classes = os.listdir(root)
        classes = sorted(classes)

        if measure == 'remove':
            classes.remove('camelyon16_0')
            classes.remove('janowczyk6_0')

        classes_tmp = []

        if generalise:
            classes = classes[len(classes) // 2:]

        self.conversion = {x: i for i, x in enumerate(classes)}

        if measure != 'random':
            for i in classes:
                for img in os.listdir(os.path.join(root, str(i))):
                    self.img_list.append(os.path.join(root, str(i), img))
        else:
            for i in classes:
                for img in os.listdir(os.path.join(root, str(i))):
                    self.dic_img[i].append(os.path.join(root, str(i), img))

            nbr_empty = 0
            to_delete = []

            while True:
                for key in self.dic_img:
                    if (not self.dic_img[key]) is False:
                        img = np.random.choice(self.dic_img[key])
                        self.dic_img[key].remove(img)
                        self.img_list.append(img)
                    else:
                        to_delete.append(key)

                for key in to_delete:
                    self.dic_img.pop(key, None)

                to_delete.clear()

                if len(self.img_list) > 1000 or len(self.dic_img) == 0:
                    break

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        return self.img_list[idx]

def test(model, dataset, db_name, extractor, measure, generalise, topsis=True):
    database = db.Database(db_name, model, True, extractor=='transformer')

    data = TestDataset(dataset, measure, generalise)

    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True,
                                         num_workers=4, pin_memory=True)

    top_1_acc = 0
    top_5_acc = 0

    dic_top5 = Counter()
    dic_top1 = Counter()

    nbr_per_class = Counter()

    ground_truth = []
    predictions = []

    t_search = 0
    t_model = 0
    t_tot = 0

    for i, image in enumerate(loader):
        t = time.time()
        if not bool(topsis):
            print('Not Topsis')
            names, _, t_model_tmp, t_search_tmp = database.search(Image.open(image[0]).convert('RGB'))
            similar = names[:5]
            w_returned = []
        
        else:
            print('Topsis')

            names, distances, t_model_tmp, t_search_tmp = database.search(Image.open(image[0]).convert('RGB'), 198)
            
            clinical_data_query, clinical_data = get_clinical_data(name_image=image[0].split('/')[-1])

            hamming_distances = get_hamming_distances(clinical_data_query, clinical_data)

            similar, w_returned = apply_topsis(hamming_distances, (names, distances[0]), topsis)
            similar = similar[:5]
        
        print(image[0], similar)
        
        t_tot += time.time() - t
        t_model += t_model_tmp
        t_search += t_search_tmp

        already_found_5 = False

        end_test = image[0].rfind("/")
        begin_test = image[0].rfind("/", 0, end_test) + 1

        nbr_per_class[image[0][begin_test: end_test]] += 1
        ground_truth.append(data.conversion[image[0][begin_test: end_test]])

        for j in range(len(similar)):
            end_retr = similar[j].rfind("/")
            begin_retr = similar[j].rfind("/", 0, end_retr) + 1
            if j == 0:
                if similar[j][begin_retr:end_retr] in data.conversion:
                    predictions.append(data.conversion[similar[j][begin_retr:end_retr]])
                else:
                    predictions.append(1000)

            if similar[j][begin_retr:end_retr] == image[0][begin_test: end_test] \
                and already_found_5 is False:
                top_5_acc += 1
                dic_top5[similar[j][begin_retr:end_retr]] += 1
                already_found_5 = True
                if j == 0:
                    dic_top1[similar[j][begin_retr:end_retr]] += 1
                    top_1_acc += 1

        # print("top 1 accuracy {}, round {}".format((top_1_acc / (i + 1)), i + 1))
        # print("top 5 accuracy {}, round {} ".format((top_5_acc / (i + 1)), i + 1))


    # print("top1:")
    # for key in sorted(dic_top1.keys()):
    #     print(key.replace("_", "\_") + " & " + str(round(dic_top1[key] / nbr_per_class[key], 2)) + "\\\\")
    # print("top5:")
    # for key in sorted(dic_top5.keys()):
    #     print(key.replace("_", "\_") + " & " + str(round(dic_top5[key] / nbr_per_class[key], 2)) + "\\\\")
    print("top-1 accuracy : ", top_1_acc / data.__len__())
    print("top-5 accuracy : ", top_5_acc / data.__len__())

    print('t_tot:', t_tot)
    print('t_model:', t_model)
    print('t_search:', t_search)

    results_infos_dict = {
        'model': extractor,
        'topsis': "False" if not bool(topsis) else (f'Entropy: {w_returned}' if topsis == [1.0, 1.0] else topsis),
        'metric': 'cosseno',
        'epochs': 100,
        'top-1 accuracy': top_1_acc / data.__len__(),
        'top-5 accuracy': top_5_acc / data.__len__(),
        't_tot': t_tot,
        't_model': t_model,
        't_search': t_search
    }
    
    import pandas as pd
    filename = 'results_all_possibilites.csv'

    # Verificar se o arquivo já existe
    file_exists = os.path.isfile(filename)

    # Criar DataFrame com base no dicionário
    df = pd.DataFrame([results_infos_dict])

    # Salvar DataFrame no arquivo CSV
    if file_exists:
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)

    import seaborn as sn
    import matplotlib.pyplot as plt
    import sklearn.metrics
    import pandas as pd
    cm = sklearn.metrics.confusion_matrix(ground_truth, predictions)
    df_cm = pd.DataFrame(cm, index=data.conversion.keys(), columns=data.conversion.keys())
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, xticklabels=True, yticklabels=True)
    #plt.show()
    plt.savefig(f'./confusion_matrix{}')

def get_clinical_data( name_image, path_csv_train="./CLINICAL/ndbufes_TaskIV_parsed_folders.csv", path_csv_test="./CLINICAL/ndbufes_TaskIV_parsed_test.csv"):
        import pandas as pd

        df_train = pd.read_csv(path_csv_train).drop(columns=['larger_size', 'TaskIV', 'folder'])
        df_test = pd.read_csv(path_csv_test).drop(columns=['larger_size', 'TaskIV', 'folder'])

        query_series = df_test[df_test['path'] == name_image].squeeze()

        clinical_data_train = [(row['path'], row.drop('path').tolist()) for _, row in df_train.iterrows()]
        clinical_data_query = (query_series['path'], query_series.drop(labels=['path']).to_list())

        return clinical_data_query, clinical_data_train
    
def get_hamming_distances(clinical_data_query, clinical_data_train):
    from scipy.spatial.distance import hamming

    name_query, clinical_vector_query = clinical_data_query[0], clinical_data_query[1]
    
    return [(name, hamming(clinical_vector_query, clinical_vector)) for name, clinical_vector in clinical_data_train]
    
def apply_topsis(hamming_distances, l2_distances, weights):
    
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
   
    #weights = [0.7, 0.5]
    criterias = [False, False]

    topsis = Topsis(topsis_distances, weights, criterias)
    w_return = topsis.calc()

    best_alternatives = topsis.rank_to_best_similarity()
    #print(best_alternatives)
    #print([topsis_data[i - 1][0] for i in best_alternatives])
    return [topsis_data[i - 1][0] for i in best_alternatives], w_return


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--num_features',
        help='number of features to extract',
        default=128,
        type=int
    )

    parser.add_argument(
        '--path',
        default='patch/val'
    )

    parser.add_argument(
        '--extractor',
        default='densenet'
    )

    parser.add_argument(
        '--dr_model',
        action="store_true"
    )

    parser.add_argument(
        '--weights',
        help='file that contains the weights of the network',
        default='weights'
    )

    parser.add_argument(
        '--db_name',
        default='db'
    )

    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int
    )

    parser.add_argument(
        '--measure',
        help='random samples from validation set <random>, remove camelyon16_0 and janowczyk6_0 <remove> or all <all>'
    )

    parser.add_argument(
        '--generalise',
        help='use only half the classes to compute the accuracy'
    )
    parser.add_argument(
        '--topsis',
        default=[],
        type=float,
        nargs='+'
    )
    args = parser.parse_args()
    print(args)
    if args.gpu_id >= 0:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    if not os.path.isdir(args.path):
        print('Path mentionned is not a folder')
        exit(-1)

    model = Model(num_features=args.num_features, name=args.weights, model=args.extractor,
                  use_dr=args.dr_model, device=device)

    test(model, args.path, args.db_name, args.extractor, args.measure, args.generalise, topsis=args.topsis)
