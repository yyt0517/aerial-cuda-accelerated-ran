"""Known-interaction attribution test; synthetic outputs stay in a temp dir."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from common import ARMS,OUT,SEEDS,write
import analyze


def main():
    # f(C,G,W)=100+10C+20G+30GW+40CG+60CGW.
    # Closed-form Shapley: C=10+40/2+60/3=50;
    # G=20+30/2+40/2+60/3=75; W=30/2+60/3=35.
    values=dict(pfq=100,ctbs=110,raw_global_equal=120,raw_global_deficit=150,
        ctbs_global_equal=170,full=260,ctbs_independent_equal=122,ctbs_independent_deficit=122)
    rows=[]
    for seed in SEEDS:
        for arm in ARMS:
            rows.append(dict(seed=seed,arm=arm,**{metric:values[arm]+seed for metric in analyze.METRICS}))
    import json
    protocol=json.loads((OUT/'protocol.json').read_text())
    with tempfile.TemporaryDirectory(prefix='dpls_attribution_check_') as temporary:
        original=analyze.OUT
        try:
            analyze.OUT=Path(temporary)
            analyze.attribution(pd.DataFrame(rows),protocol)
            result=pd.read_csv(Path(temporary)/'shapley_summary.csv')
            for metric in analyze.METRICS:
                sub=result[result.metric==metric].set_index('factor')
                for factor,expected in dict(CTBS=50,GLOBAL_SEARCH=75,DEFICIT=35).items():
                    np.testing.assert_allclose(sub.loc[factor,'mean'],expected,rtol=0,atol=1e-12)
                np.testing.assert_allclose(sub['mean'].sum(),160,rtol=0,atol=1e-12)
            interactions=pd.read_csv(Path(temporary)/'interaction_summary.csv')
            x=interactions[interactions.metric=='complete_payload_mbps'].set_index('interaction')
            assert x.loc['ctbs_x_search_equal','mean']==40
            assert x.loc['ctbs_x_deficit_under_search','mean']==60
            contrasts=pd.read_csv(Path(temporary)/'contrasts.csv')
            null=contrasts[(contrasts.added=='ctbs_independent_deficit')&(contrasts.reference=='ctbs_independent_equal')&(contrasts.metric=='complete_payload_mbps')].iloc[0]
            assert null['mean']==0 and null.p_holm_primary==1
        finally:analyze.OUT=original
    result=analyze.summary([1,2,3,4,5])
    assert result['n']==5 and result['mean']==3
    np.testing.assert_allclose([result['ci95_low'],result['ci95_high']],[1.03675684,4.96324316],atol=1e-8,rtol=0)
    write(OUT/'analysis_checks.json',dict(passed=True,synthetic_known_shapley_all_metrics=True,
        known_interactions=True,contribution_sum_exact=True,zero_effect_holm_p_one=True,
        paired_t_interval_known_result=True,no_synthetic_results_in_main_output=True))
    print('ANALYSIS_ARITHMETIC_CHECKED')


if __name__=='__main__':main()
