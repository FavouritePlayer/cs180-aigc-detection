import unittest
from detection_metrics import auroc,bootstrap_auc
class DetectionMetricsTest(unittest.TestCase):
    def test_fixed_direction(self):
        self.assertEqual(auroc([0,0,1,1],[1,2,3,4]),1)
        self.assertEqual(auroc([0,0,1,1],[4,3,2,1]),0)
    def test_ties(self):
        self.assertEqual(auroc([0,1],[2,2]),.5)
    def test_perfect_bootstrap(self):
        self.assertEqual(bootstrap_auc([0,0,1,1],[1,2,3,4],replicates=20),[1,1])
if __name__=='__main__':unittest.main()
