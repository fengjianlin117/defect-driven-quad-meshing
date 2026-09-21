import unittest
from types import SimpleNamespace
import numpy as np
from coordination import constraint_operation,common_metrics,failure_removal
from conflict_selection import EqualityAudit

def row(edge,value=1):return dict(vertices=list(edge),length_h=1.,layer='main',surface_loss=value,unaligned_fraction=value,surface_distance_p95_h=value*.2)
def audit():
    mesh=SimpleNamespace(vertices=np.array([[0.,0.,0.],[2.,0.,0.],[1.,.1,0.]]),faces=np.array([[0,1,2]]))
    return mesh,EqualityAudit(mesh,np.zeros((3,2)),mesh.faces,np.array([[1.,0.,0.]]),np.array([[0.,1.,0.]]))

class CoordinationTests(unittest.TestCase):
    def test_self_conflicting_group_not_treated_as_replaceable(self):
        mesh,a=audit();groups={'bad':{(0,1),(0,2)}}
        r=constraint_operation(groups,[],[row((0,1)),row((0,2))],a)
        self.assertIsNone(r['proposal']);self.assertEqual(r['screening_calls'],1)
        self.assertEqual(r['trace'][0]['status'],'within_group_fixed_frame_conflict')
    def test_actual_union_conflict_has_single_replacement(self):
        mesh,a=audit();groups={'a':{(0,1)},'b':{(0,2)}}
        r=constraint_operation(groups,['a'],[row((0,1),.1),row((0,2))],a)
        self.assertEqual(r['proposal']['remove'],'a');self.assertEqual(r['proposal']['groups'],['b'])
        self.assertEqual(r['proposal']['displaced_edges'],[[0,1]])
    def test_no_diagnosed_need_does_not_add_for_small_continuous_loss(self):
        mesh,a=audit();r=row((0,1),0.);r['surface_loss']=.05
        result=constraint_operation({'a':{(0,1)}},[],[r],a)
        self.assertIsNone(result['proposal']);self.assertEqual(result['screening_calls'],0)
    def test_screening_budget_is_hard(self):
        mesh,a=audit();result=constraint_operation({'a':{(0,1)},'b':{(0,2)}},['a'],[row((0,1)),row((0,2))],a,max_screenings=1)
        self.assertLessEqual(result['screening_calls'],1);self.assertIsNone(result['proposal'])
    def test_shared_reference_regression_not_hidden_by_net_gain(self):
        base={'edge_defects':[row((0,1),.8),row((0,2),.0)]}
        after={'edge_defects':[row((0,1),.0),row((0,2),.2)]}
        r=common_metrics(after,base,set());self.assertAlmostEqual(r['common_positive_regression'],.2)
        self.assertEqual(r['newly_deficient_edges'],1)
    def test_reference_identity_is_mandatory(self):
        with self.assertRaises(ValueError):common_metrics({'edge_defects':[row((0,1))]},{'edge_defects':[row((0,2))]},set())

if __name__=='__main__':unittest.main()
