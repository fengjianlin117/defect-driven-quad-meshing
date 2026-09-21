import unittest
import numpy as np
from cad_dimension_proxies import circle
class DimensionTests(unittest.TestCase):
    def test_circle_radius_and_center_in_tilted_plane(self):
        t=np.linspace(0,2*np.pi,32,endpoint=False);a=np.array([1.,0.,0.]);b=np.array([0.,.6,.8]);center=np.array([3.,-1.,2.])
        points=center+2.5*np.cos(t)[:,None]*a+2.5*np.sin(t)[:,None]*b
        fit=circle(points);self.assertAlmostEqual(fit['radius'],2.5,places=12)
        np.testing.assert_allclose(fit['center'],center,atol=1e-12)
        self.assertLess(fit['max_plane_residual'],1e-12);self.assertLess(fit['max_radial_residual'],1e-12)
    def test_collinear_points_not_given_radius(self):
        self.assertIsNone(circle(np.c_[np.arange(8),np.zeros(8),np.zeros(8)]))
    def test_non_circular_loop_has_reported_residual(self):
        t=np.linspace(0,2*np.pi,32,endpoint=False);fit=circle(np.c_[2*np.cos(t),np.sin(t),np.zeros(32)])
        self.assertGreater(fit['max_radial_residual'],.1)
if __name__=='__main__':unittest.main()
