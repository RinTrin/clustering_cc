import os
import pandas as pd
import requests
import zipfile
import io
import csv
import pickle
import json
from tqdm import tqdm
import numpy as np
import gc
import convert_config


def get_data(opt):
    # brf_sum_text_folder = "/content/brf_sum_text_folder/"
    brf_sum_text_folder = os.path.join(opt.data.home_dir, "brf_sum_text")
    os.makedirs(brf_sum_text_folder, exist_ok=True)
    for year in range(2005, 2022):
        if year >= 2006:
            break
        json_path = os.path.join(brf_sum_text_folder, f"{str(year)}.json")
        if os.path.exists(json_path):
            print(f"year :{year} of json already exists!!")
            continue
        else:
            print(f"year :{year} isn't exist. Let's make")
        url_path_each = f"https://s3.amazonaws.com/data.patentsview.org/pregrant_publications/brf_sum_text_{str(year)}.tsv.zip"
        response = requests.get(url_path_each, stream=True)
        if "200" == str(response.status_code):
            print(f"year :{year} is succesfully accessed")
        else:
            print("something wrong with requests")
            break

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            file_name = f"brf_sum_text_{str(year)}.tsv"
            df_reader = pd.read_csv(
                zf.open(file_name),
                delimiter="\t",
                quoting=csv.QUOTE_NONNUMERIC,
                chunksize=1000,
                dtype={
                    "id": np.object,
                    "document_number": np.object,
                    "text": np.object,
                },
            )
            df = pd.concat((r for r in df_reader), ignore_index=True)
            json_dict = {}
            for idx in tqdm(range(len(df.index))):
                row = df.iloc[idx, :]
                each_dict = {}
                for label in ["id", "document_number", "text"]:
                    each_dict[label] = row[label]
                json_dict[idx] = each_dict
            with open(json_path, "w") as jf:
                json.dump(json_dict, jf)
            print(df.describe(exclude=[np.number]))
            del df
            del df_reader
            del json_dict
            gc.collect()
            print(
                f"size of the json file is : {os.path.getsize(json_path) / 1024**3:.04f} GB"
            )
            print(f"year :{year} finished")


if __name__ == "__main__":
    config = convert_config.convert_config(
        path="/home/kento/tomita/my_ML/clustering_cc/patent_clustering/config/base.yaml"
    )
    get_data(config)
