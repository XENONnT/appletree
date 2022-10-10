import os

from appletree import ComponentFixed
from appletree.share import DATAPATH


class AC(ComponentFixed):
    file_name = os.path.join(DATAPATH, 'AC_Rn220.pkl')
    rate_name = 'ac_rate'
    norm_type = 'on_pdf'
