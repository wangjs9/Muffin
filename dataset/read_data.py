""" This file is used to read dataset contained in training-dataset.with-ids.tsv.
    This dataset is download from ``https://github.com/davidjurgens/support/tree/master``.
    There are four attributes for each response: ``agreement``, ``offensiveness``, ``politeness``, ``support``.
 """

import pandas as pd
import numpy as np

FILE_NAME = "training-dataset.with-ids.tsv"


def read_data():
    agreement_list = []
    supportness_list = []
    data = pd.read_csv(FILE_NAME, sep='\t', header=0, index_col=None, encoding="UTF-8")
    print(data.head())
    for idx, row in data.iterrows():
        # print(row)
        agreement_list.append(row['agreement'])
        supportness_list.append(row['support'])
    print("Agreement List:")
    print(np.mean(agreement_list), np.median(agreement_list), max(agreement_list), min(agreement_list))
    print("Supportness List:")
    print(np.mean(supportness_list), np.median(supportness_list), max(supportness_list), min(supportness_list))


read_data()
