import os

from appletree import ComponentFixed
from appletree.share import DATAPATH

class AC(ComponentFixed):
    file_name = os.path.join(DATAPATH, 'data_XENONnT_Rn220_v8_strax_v1.2.2_straxen_v1.7.1_cutax_v1.9.0.csv')
    rate_par_name = 'ac_rate'
    norm_type = 'on_pdf'
