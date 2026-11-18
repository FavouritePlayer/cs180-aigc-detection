import unittest
import numpy as np
from matched_metrics import matched_summary
class MatchedMetricsTest(unittest.TestCase):
    def test_perfect_and_paired_constant(self):
        s=matched_summary([0,0],[[1,2],[1,2]],replicates=20)
        self.assertEqual(s['auroc'],1);self.assertEqual(s['auroc_95pct_caption_cluster_bootstrap'],[1,1]);self.assertEqual(s['paired_difference_95pct_caption_cluster_bootstrap'],[1.5,1.5])
    def test_fixed_reverse_direction(self):
        s=matched_summary([3,3],[[1,2],[1,2]],replicates=20)
        self.assertEqual(s['auroc'],0);self.assertEqual(s['paired_mean_difference'],-1.5)
    def test_macro_identity_and_ties(self):
        s=matched_summary([0,1],[[0,2],[1,3]],replicates=20)
        self.assertAlmostEqual(s['auroc'],np.mean(s['per_generator_auroc']));self.assertEqual(s['per_generator_auroc'][0],.5)
    def test_shape_validation(self):
        with self.assertRaises(ValueError):matched_summary([0,1],[[2,3]],replicates=20)
if __name__=='__main__':unittest.main()
