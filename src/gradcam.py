# src/gradcam.py

import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def forward_hook(module, inp, out):
            self.activations = out

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: shape (1, C, H, W)
        returns: 2D numpy array (cam) normalized to [0,1]
        """
        self.model.zero_grad()
        out = self.model(input_tensor)

        if class_idx is None:
            class_idx = out.argmax(dim=1).item()

        score = out[:, class_idx]
        score.backward()

        grads = self.gradients        # (N, C, H, W)
        acts  = self.activations      # (N, C, H, W)

        weights = grads.mean(dim=(2, 3), keepdim=True)   # GAP over H,W
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (N,1,H,W)
        cam = F.relu(cam)

        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam  # still low-res; will resize later
# ---------------------------