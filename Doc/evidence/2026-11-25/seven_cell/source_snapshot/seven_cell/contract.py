"""Same integer objective/key constraints, widened UE dependency mask."""
import numpy as np
SCALE=1_000_000

def check_contract(p):
    if p.U+p.P>127 or p.P*(p.U+3)**p.C>=2**63:raise ValueError('Unsupported sparse-mask/key dimensions')
    if not np.array_equal(p.queue,np.floor(p.queue)):raise ValueError('Integer byte queues required')
    bound=sum(int(np.ceil(q*w*SCALE))+1 for q,w in zip(p.queue,p.weights))
    if bound>=np.iinfo(np.int64).max//4:raise OverflowError('Integer objective bound is unsafe')

