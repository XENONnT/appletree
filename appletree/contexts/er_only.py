from appletree.context import Context
from appletree.utils import get_file_path


class ContextRn220(Context):
    """A specified context for ER response by Rn220 fit"""

    def __init__(self):
        """Initialization"""
        config = get_file_path('rn220_sr0.json')
        super().__init__(config)


class ContextRn220Ar37(Context):
    """A specified context for ER response by Rn220 & Ar37 combined fit"""

    def __init__(self):
        """Initialization"""
        config = get_file_path('rn220_ar37_sr0.json')
        super().__init__(config)
