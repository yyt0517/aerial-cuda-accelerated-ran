import numpy as np
from .fast_phy import FastPhy,CausalPhy

class VerifiedPhy(FastPhy):
    def __init__(self,config):
        super().__init__(config)
        self.audit_calls=0
    def evaluate(self,actions):
        prediction=super().evaluate(actions)
        if self.snapshot['tti'] in [4000,4001,4002] or self.snapshot['tti']%1000==0:
            reference=CausalPhy.evaluate(self,actions)
            for key in prediction:
                np.testing.assert_array_equal(prediction[key],reference[key],err_msg=key)
            self.audit_calls+=1
        return prediction
