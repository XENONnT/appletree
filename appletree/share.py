_cached_configs = dict()
_cached_functions = dict()


def set_global_config(configs):
    """Set new global configuration options

    :param configs: dict, configuration file name or dictionary
    """
    from appletree.utils import get_file_path

    for k, v in configs.items():
        if isinstance(v, (float, int, list)):
            _cached_configs.update({k: v})
        elif isinstance(v, str):
            file_path = get_file_path(v)
            _cached_configs.update({k: file_path})
        else:
            raise NotImplementedError
