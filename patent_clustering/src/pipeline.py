from cgitb import text
import os
import numpy as np
import pandas as pd
import json
import torch
import convert_config
import check_similarity

from make_feature import make_feature
from data_utils import add_subsection, extract_json


def pipeline(opt):

    if opt.test.debug:
        print("loading brief sum text json data")
        with open(opt.test.brf_sum_text_path, "r") as jf:
            text_data = json.load(jf)
        # with open(opt.data.brf_sum_text_path, "r") as jf:
        #     text_data = json.load(jf)
        print("load data finish")
        # extract_json(opt, text_data, 2021)

        # add_subsection(opt, text_data) # <- うまくいかない

        # njk
        features_dict = make_feature(opt, text_data=text_data)

    else:
        # real dev
        # # using only 2021 data for test
        print("loading brief sum text json data")
        with open(opt.data.brf_sum_text_path, "r") as jf:
            text_data = json.load(jf)
        print("load data finish")

        features_dict = make_feature(opt, text_data=text_data)

    check_similarity.check_similarity(opt, features_dict)


if __name__ == "__main__":
    config = convert_config.convert_config(
        path="/home/kento/tomita/my_ML/clustering_cc/patent_clustering/config/base.yaml"
    )
    pipeline(config)
