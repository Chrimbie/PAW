"""Regression tests for experimental AGR. Run: python -m unittest discover -s experiments/agr -p 'test_*.py'"""
import unittest
import torch
from agr import adaptive_gradient_recovery


class AGRTests(unittest.TestCase):
    def test_forward_matches_original(self):
        z = torch.tensor([-9., -1.25, -.3, 0., .3, 1.25, 9.])
        expected = torch.tanh(z.clamp(-1.25, 1.25))
        self.assertTrue(torch.equal(adaptive_gradient_recovery(z), expected))

    def test_directional_recovery(self):
        z = torch.tensor([4., 4., -4., -4.], requires_grad=True)
        upstream = torch.tensor([1., -1., -1., 1.])
        adaptive_gradient_recovery(z).backward(upstream)
        self.assertGreater(z.grad[0].item(), 0.)
        self.assertEqual(z.grad[1].item(), 0.)
        self.assertLess(z.grad[2].item(), 0.)
        self.assertEqual(z.grad[3].item(), 0.)

    def test_inside_matches_tanh_derivative(self):
        z = torch.tensor([-.5, 0., .5], requires_grad=True)
        adaptive_gradient_recovery(z).sum().backward()
        self.assertTrue(torch.allclose(z.grad, 1-torch.tanh(z).square()))

    def test_invalid_limit(self):
        with self.assertRaises(ValueError):
            adaptive_gradient_recovery(torch.tensor([0.]), limit=0)


if __name__ == "__main__":
    unittest.main()
