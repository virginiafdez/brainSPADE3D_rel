import pickle
import torch
class UnparsedOptions():
    def __init__(self, load_path):
        self.load_path = load_path
        self.initialized = False

    def initialize(self):

        opt = pickle.load(open(self.load_path + 'opt.pkl', 'rb'))
        opt.semantic_nc = opt.label_nc + \
                          (1 if opt.contain_dontcare_label else 0) + \
                          (0)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        assert len(opt.gpu_ids) == 0 or opt.batchSize % len(opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batchSize, len(opt.gpu_ids))

        self.initialized = True
        self.opt = opt
        return self.opt