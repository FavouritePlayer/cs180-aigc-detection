import unittest
import numpy as np
from metrics import depth_metrics, normal_metrics

class MetricsTest(unittest.TestCase):
    def test_metric_scale_and_mask(self):
        gt=np.array([[1.,2.],[0.,100.]])
        pred=np.array([[2.,4.],[np.nan,100.]])
        m=depth_metrics(gt,pred,np.array([[1,1],[0,0]]))
        self.assertEqual(m['abs_rel'],1.)
        self.assertEqual(m['median_aligned_abs_rel'],0.)
        self.assertEqual(m['pixels'],2)
    def test_bad_prediction_is_not_silently_excluded(self):
        with self.assertRaises(ValueError):depth_metrics(np.ones((2,2)),np.zeros((2,2)),np.ones((2,2)))
    def test_oriented_normals_and_missing_gt(self):
        gt=np.array([[[0.,0.,1.],[0.,0.,0.]]])
        pred=-gt
        m=normal_metrics(gt,pred,np.ones((1,2),dtype=bool))
        self.assertEqual(m['mean_angle'],180.)
        self.assertEqual(m['pixels'],1)

if __name__=='__main__':unittest.main()
