import os

from appletree import ComponentFixed
from appletree.share import DATAPATH

class AC(ComponentFixed):
    file_name = os.path.join(DATAPATH, 'ac_radon_0616.pkl')
    rate_par_name = 'ac_rate'
    norm_type = 'on_pdf'
