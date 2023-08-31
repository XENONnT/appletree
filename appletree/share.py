import json


class RecordingDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accessed_keys = set()  # To store the accessed keys

    def __getitem__(self, key):
        self.accessed_keys.add(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        self.accessed_keys.add(key)
        return super().__setitem__(key, value)

    def __repr__(self):
        return json.dumps(self, indent=4)

    def __str__(self):
        return self.__repr__()

    def clear(self):
        super().clear()
        self.accessed_keys.clear()


_cached_configs = RecordingDict()
_cached_functions = dict()


def set_global_config(configs):
    """Set new global configuration options.

    :param configs: dict, configuration file name or dictionary

    """
    from appletree.utils import get_file_path

    for k, v in configs.items():
        if isinstance(v, (float, int, list)):
            _cached_configs.update({k: v})
        elif isinstance(v, str):
            file_path = get_file_path(v)
            _cached_configs.update({k: file_path})
        elif isinstance(v, dict):
            file_path_dict = dict()
            for kk, vv in v.items():
                if isinstance(vv, (float, int, list)):
                    file_path_dict[kk] = vv
                elif isinstance(vv, str):
                    file_path_dict[kk] = get_file_path(vv)
            _cached_configs.update({k: file_path_dict})
        else:
            raise NotImplementedError
