import unittest
from coordinator_v5 import strict_decision

def row(name,d=10,strict=True):
    return dict(id=name,process_success=True,basic_output_pass=True,strict_output_pass=strict,quads=500,
        all_main_deficit=d,symmetric_rms_h=.02,protected_deficit=1.)

class GeometryDecisionTests(unittest.TestCase):
    def test_better_but_intersecting_candidate_rejected(self):
        r=strict_decision([row('baseline'),row('improved',1,False)],2048)
        self.assertEqual(r['recommended_id'],'baseline');self.assertEqual(r['eligible_ids'],['baseline'])
    def test_missing_check_rejects_and_invalid_baseline_not_assumed_safe(self):
        a=row('baseline',strict=False);b=row('candidate',1);del b['strict_output_pass']
        self.assertIsNone(strict_decision([a,b],2048)['recommended_id'])
    def test_original_protection_not_relaxed(self):
        a=row('candidate',1);a['protected_deficit']=1.01
        self.assertEqual(strict_decision([row('baseline'),a],2048)['recommended_id'],'baseline')

if __name__=='__main__':unittest.main()
