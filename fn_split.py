# coding: utf-8

# # 說明
# 整理切好分批為 training / testing sets, 共切成 5 份

import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
import json
from sklearn.utils import shuffle
import sys

def split():
    ROOT_PATH = os.path.dirname(os.getcwd())
    SRC_FOLDER = f'{ROOT_PATH}/fma_preprocessing'

    with open(f'{ROOT_PATH}/dylanchi/category_label_mapping.json', 'r') as f:
        categoryMapping = json.load(f)
    iptPath = []
    category = []
    reverseCategoryMapping = {v: k for k, v in categoryMapping.items()}
    for dirpath, _, filenames in os.walk(SRC_FOLDER):
        for f in filenames:
            cat = dirpath.split('/')[4]
            category.append(reverseCategoryMapping[cat])
            iptPath.append(f'../{dirpath[2:]}/{f}')
    dfNew = pd.concat([pd.Series(category), pd.Series(iptPath)], axis=1)
    dfNew.columns = ['category', 'path']
    dfAll = shuffle(dfNew)
    y = dfAll['category'].tolist()
    input_ = dfAll['path'].tolist()

    return input_, y