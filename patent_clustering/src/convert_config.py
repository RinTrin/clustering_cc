import yaml


def convert_config(path=None):

    if "yaml" in path:
        pass
    else:
        print("=====================")
        print("path type is not yaml")
        print("=====================")

    with open(path, "r") as yml:
        config = yaml.safe_load(yml)
    new_config = {}
    for key, value in config.items():
        new_config[key] = dict_dot_notation(value)
    dot_config = dict_dot_notation(new_config)
    return dot_config


class dict_dot_notation(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


if __name__ == "__main__":
    convert_config()
