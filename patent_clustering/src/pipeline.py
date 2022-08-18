import os
import numpy as np
import pandas as pd
import json
import torch
import convert_config
import check_similarity

from make_feature import make_feature


def pipeline(opt):

    # using only 2005 data for test
    with open(opt.data.brf_sum_text_path, "r") as jf:
        text_data = json.load(jf)

    features_dict = make_feature(text_data)

    check_similarity(features_dict)


if __name__ == "__main__":
    config = convert_config.convert_config(
        path="/home/kento/tomita/my_ML/clustering_cc/patent_clustering/config/base.yaml"
    )
    pipeline(config)
