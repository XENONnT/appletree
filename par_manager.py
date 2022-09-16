import numpy as np

class par_manager():
    def __init__(self):
        self._parameter_dict = {
            "fano": 0.059,
            "W": 0.013509665661431896, 
            "Nex/Ni": 0.08237994367314523, 
            "g1": 0.15008491987690997, 
            "g2": 16.52840260432468, 
            "p_dpe": 0.2873015066181266, 
            "py0": 0.12644250072199228, 
            "py1": 43.12392476032283, 
            "py2": -0.30564651066249543, 
            "py3": 0.937555814189728, 
            "py4": 0.5864910020458629, 
            "rf0": 0.029414125811261564, 
            "rf1": 0.2571929264699089,
            "field": 23.,
            "drift_velocity": 0.0677,
            "gas_gain": 31.3,
            "s2_threshold": 500,
        }
        
    def __eq__(self):
        pass
    
    def __getitem__(self, keys):
        if isinstance(keys, list):
            return np.array([self._parameter_dict[key] for key in keys])
        
        elif isinstance(keys, str):
            return self._parameter_dict[keys]
        
        else:
            raise ValueError("keys must be a str or a list of str!")
    
    def get_parameter(self, keys):
        """
        Return parameter values.
        
        keys : Parameter names. Can be a single str, or a list of str.
        """
        all_exist, not_exist = self.check_parameter_exist(keys, return_not_exist=True)
        assert all_exist, "%s not found!"%not_exist
        
        return self.__getitem__(keys)
    
    def set_parameter(self, keys, vals=None):
        """
        Set parameter values.
        
        keys : Parameter names. Can be a single str, or a list of str, or a dict.
               If str, vals must be int or float.
               If list, vals must have the same length.
               If dict, vals will be overwritten as keys.values() and ignore vals.
        vals : values to be set.       
        """
        all_exist, not_exist = self.check_parameter_exist(keys, return_not_exist=True)
        assert all_exist, "%s not found!"%not_exist
            
        if isinstance(keys, list):
            assert len(keys)==len(vals), "keys must have the same length as vals!"
            for key, val in zip(keys, vals):
                self._parameter_dict[key] = val
                
        elif isinstance(keys, dict):
            self.set_parameter(list(keys.keys()), keys.values())
            
        elif isinstance(keys, str):
            assert isinstance(vals, (float, int)), "if there is only one key, val must be either float or int!"
            self._parameter_dict[keys] = vals
            
        else:
            raise ValueError("keys must be a str or a list of str!")
    
    def check_parameter_exist(self, keys, return_not_exist=False):
        """
        Check whether the keys exist in parameters.
        
        keys             : Parameter names. Can be a single str, or a list of str.
        return_not_exist : If False, function will return a bool if all keys exist. If True, function will additionally return the not existing list of keys.
        """
        if isinstance(keys, list):
            not_exist = []
            for key in keys:
                if not key in self._parameter_dict:
                    not_exist.append(key)
            all_exist = (not_exist==[])
            if return_not_exist:
                return (all_exist, not_exist)
            else:
                return (all_exist)
            
        elif isinstance(keys, str):
            if return_not_exist:
                return (keys in self._parameter_dict, keys)
            else:
                return (keys in self._parameter_dict)
            
        elif isinstance(keys, dict):
            return self.check_parameter_exist(list(keys.keys()), return_not_exist)
        
        else:
            raise ValueError("keys must be a str or a list of str!")
          
    def get_all_parameter(self):
        """
        Return all parameters as a dict.     
        """
        return self._parameter_dict
    
    