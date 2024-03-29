from __future__ import print_function
from __future__ import division
from . import _C

import torch
from fuzzytools.files import create_dir, PFile
from copy import copy, deepcopy

###################################################################################################################################################

def override(func): return func # trick
class FTFile(PFile):
    def __init__(self, filedir,
        model=None,
        lmonitors=None,
        ):
        file = None
        if not model is None:
            file = {
                'state_dict':deepcopy(model.state_dict()), # it needs to be deepcopy
                'lmonitors':{lmonitor.name:lmonitor.get_save_dict() for lmonitor in lmonitors},
                }
        super().__init__(filedir,
            file,
            )

    @override
    def _save(self,
        copy_filedirs=[],
        ):
        filedirs = [self.filedir]+copy_filedirs
        for filedir in filedirs:
            torch.save(self.file, filedir)
        self.last_state = 'saved'
        return