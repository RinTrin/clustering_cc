import os
import json
import csv
from tqdm import tqdm
import pandas as pd

"""
this python file is for garbage of utils for arranging data
"""


def add_subsection(opt, brf_sum_text):

    cpc_current_path = os.path.join(opt.data.home_dir, "connecting_csv/cpc_current.tsv")
    current_df = pd.read_csv(
        cpc_current_path, delimiter="\t", quoting=csv.QUOTE_NONNUMERIC
    )
    # print(current_df.columns)
    """Index(['Cpc"uuid"', 'patent_id', 'section_id', 'subsection_id', 'group_id',
       'subgroup_id', 'category', 'sequence'],"""

    print("load cpc current data finish")
    print(current_df['Cpc"uuid"'].iloc[:10])
    print(list(current_df['Cpc"uuid"'])[:10])
    print(len(list(current_df['Cpc"uuid"'])))
    cdf_p = list(current_df["patent_id"])
    # print([:10])
    print(min(cdf_p))
    # bn

    new_brf_sum_text_data = {}
    for idx, data_dict in tqdm(brf_sum_text.items()):
        print(data_dict["uuid"], type(data_dict["uuid"]))
        print(data_dict["uuid"] in list(current_df['Cpc"uuid"']))
        # print(type(list(current_df['Cpc"uuid"'])[0]))
        # print(
        #     "patent_id",
        #     data_dict["patent_id"],
        #     data_dict["patent_id"] in list(current_df["patent_id"]),
        #     type(data_dict["patent_id"]),
        #     type(list(current_df["patent_id"])[1]),
        # )
        nm,
        # subsection_id = current_df[current_df['Cpc"uuid"'] == data_dict["uuid"]][
        #     "subsection_id"
        # ]
        # print(subsection_id)
        # bnjk
        if int(idx) == 100:
            break
    fgthjz


def extract_json(opt, text_data, year, num=10000):
    test_text_data = {}
    for idx, text_dict in text_data.items():
        if int(idx) >= num:
            break
        test_text_data[int(idx)] = text_dict
    with open(
        f"/home/kento/tomita/my_ML/clustering_cc/patent_clustering/data/brf_sum_text/test_{str(year)}.json",
        "w",
    ) as jf:
        json.dump(test_text_data, jf)
