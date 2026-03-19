import yaml

def read_config_file(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    # flatten the dictionary such that <key1.key2> becomes <key1_key2>
    flat_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_config[f"{key}_{sub_key}"] = sub_value
        else:
            flat_config[key] = value
    return flat_config