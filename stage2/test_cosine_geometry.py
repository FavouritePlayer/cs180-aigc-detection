import unittest
import numpy as np
from cosine_geometry import forward_normals,cosine_residual
class CosineGeometryTest(unittest.TestCase):
    def test_flat_depth_orientation_and_border(self):
        k=np.array([[100,0,2],[0,100,2],[0,0,1.]])
        n,v=forward_normals(np.full((5,6),2.),k)
        np.testing.assert_allclose(n[v],np.tile([0,0,1],(20,1)))
        self.assertEqual(v.sum(),20);self.assertFalse(v[-1].any());self.assertFalse(v[:,-1].any())
    def test_slanted_plane_and_scale_invariance(self):
        y,x=np.mgrid[:7,:8];k=np.array([[100,0,3],[0,100,3],[0,0,1.]])
        z=2/(1-.2*(x-3)/100-.1*(y-3)/100)
        n,v=forward_normals(z,k);expected=np.array([.2,-.1,1]);expected/=np.linalg.norm(expected)
        np.testing.assert_allclose(n[v],np.tile(expected,(v.sum(),1)),atol=1e-6)
        m,w=forward_normals(7*z,k);np.testing.assert_allclose(n[v],m[w],atol=1e-6)
    def test_residual_endpoints(self):
        a=np.tile([0.,0.,1.],(3,1));b=np.array([[0,0,1],[1,0,0],[0,0,-1.]])
        np.testing.assert_allclose(cosine_residual(a,b),[0,1,2])
    def test_invalid_stencil_is_excluded(self):
        z=np.ones((3,3));z[0,1]=np.nan
        _,v=forward_normals(z,np.eye(3));self.assertFalse(v[0,0]);self.assertFalse(v[0,1]);self.assertTrue(v[1,1])
if __name__=='__main__':unittest.main()
