import unittest,copy
from protection_feedback import repair_edges,constraint_repair

def row(vertices=(0,1),surface=.02):
    return dict(vertices=list(vertices),layer='main',length_h=1.,surface_loss=surface,unaligned_fraction=0.,surface_distance_p95_h=.01)

class PassAudit:
    def check(self,edges):return dict(admissible=True)

class TriggerTests(unittest.TestCase):
    def test_below_diagnostic_but_breaks_protection(self):
        b=[row()];p=[row(surface=.08)];required,audit=repair_edges(p,b,set())
        self.assertEqual(required,{(0,1)});self.assertEqual(audit['original_deficient_edges'],[])
        self.assertTrue(audit['aggregate_protection_failed'])
        self.assertEqual(repair_edges(p,b,set(),False)[0],set())
    def test_unchanged_or_improved_not_forced(self):
        self.assertEqual(repair_edges([row(surface=.01)],[row()],set())[0],set())
        self.assertEqual(repair_edges([row(surface=.02000000001)],[row()],set())[0],set())
    def test_selected_edges_not_protected(self):
        self.assertEqual(repair_edges([row(surface=.08)],[row()],{(0,1)})[0],set())
    def test_aggregate_rule_not_silently_replaced(self):
        b=[row(),row((1,2),.5)];p=[row(surface=.08),row((1,2),.1)]
        self.assertEqual(repair_edges(p,b,set())[0],set())
    def test_reference_identity_and_immutable_records(self):
        b=[row()];p=[row(surface=.08)];original=copy.deepcopy(p)
        answer=constraint_repair({'g':{(0,1)}},[],p,PassAudit(),b,set())
        self.assertEqual(answer['proposal']['groups'],['g']);self.assertEqual(p,original)
        with self.assertRaises(ValueError):repair_edges([row((1,2))],b,set())

if __name__=='__main__':unittest.main()
