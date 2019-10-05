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

def split(srcPath):
    mappingFile = '/home/jovyan/at082-group23/dylanchi/category_label_mapping.json'
    with open(mappingFile, 'r') as f:
        categoryMapping = json.load(f)
    iptPath = []
    category = []
    reverseCategoryMapping = {v: k for k, v in categoryMapping.items()}
    for dirpath, _, filenames in os.walk(srcPath):
        for f in filenames:
            dirAry = dirpath.split('/')
            cat = dirAry[4]
            category.append(reverseCategoryMapping[cat])
            iptPath.append(f'../{dirAry[2]}/{dirAry[3]}/{dirAry[4]}/{f}')
    dfNew = pd.concat([pd.Series(category), pd.Series(iptPath)], axis=1)
    dfNew.columns = ['category', 'path']
    dfAll = shuffle(dfNew)
    y = dfAll['category'].tolist()
    input_ = dfAll['path'].tolist()

    return input_, y