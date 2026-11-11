import unittest
import numpy as np
from metrics import depth_metrics, normal_metrics, depth_to_normals

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
    def test_front_plane_and_scaled_depth_normals(self):
        d=np.full((12,16),3.)
        n,valid=depth_to_normals(d,intrinsics=(10,10,8,6))
        self.assertTrue(np.allclose(n[valid],[0,0,1]))
        n2,v2=depth_to_normals(d*7,intrinsics=(10,10,8,6))
        self.assertTrue(np.allclose(n[valid],n2[v2]))
    def test_oriented_normals_and_missing_gt(self):
        gt=np.array([[[0.,0.,1.],[0.,0.,0.]]])
        pred=-gt
        m=normal_metrics(gt,pred,np.ones((1,2),dtype=bool))
        self.assertEqual(m['mean_angle'],180.)
        self.assertEqual(m['pixels'],1)

if __name__=='__main__':unittest.main()
