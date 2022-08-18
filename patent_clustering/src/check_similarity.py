import os
import convert_config


def check_similarity(opt):
    pass


if __name__ == "__main__":
    config = convert_config.convert_config(
        path="/home/kento/tomita/my_ML/clustering_cc/patent_clustering/config/base.yaml"
    )
    check_similarity(config)
