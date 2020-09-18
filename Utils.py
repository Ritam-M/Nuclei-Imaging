def str2bool(v):
    if v.lower() in ['true','1']:
        return True
    elif v.lower() in ['false','0']:
        return False
    else:
        return argparse.ArgumentTypeError('Boolean Value Expected')
    
def count_params(model):
    return(sum(p.numel()) for p in model.parameters() if p.requires_grad)

class AverageMeter(object):
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count+=n
        self.avg = self.sum/self.count
