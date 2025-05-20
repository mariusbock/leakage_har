# ------------------------------------------------------------------------
# Differential Privacy taken from the Breaching repository: https://github.com/JonasGeiping/breaching   
# ------------------------------------------------------------------------
# Adaption by: Marius Bock and Maximilian Hopp
# E-Mail: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------

import torch
import numpy as np

class DPrivacy():
    def __init__(self, multiplier = None, clip = None):
        self.multiplier = multiplier
        self.clip = clip
        
    def addNoise(self, gradient):
        return gradient + torch.normal(0, self.multiplier, gradient.shape)
    
    def clipGradient(self, gradient):
        return torch.clamp(gradient, -self.clip, self.clip)
     
class BreachDP():
    def __init__(self, local_diff_privacy, setup):
        self.defense_repr = []
        """Initialize generators for noise in either gradient or input."""
        self.local_diff_privacy = local_diff_privacy
        if local_diff_privacy["gradient_noise"] > 0.0:
            loc = torch.as_tensor(0.0, **setup)
            scale = torch.as_tensor(local_diff_privacy["gradient_noise"], **setup)
            if local_diff_privacy["distribution"] == "gaussian":
                self.generator = torch.distributions.normal.Normal(loc=loc, scale=scale)
            elif local_diff_privacy["distribution"] == "laplacian":
                self.generator = torch.distributions.laplace.Laplace(loc=loc, scale=scale)
            else:
                raise ValueError(f'Invalid distribution {local_diff_privacy["distribution"]} given.')
            self.defense_repr.append(f'Defense: Local {local_diff_privacy["distribution"]} gradient noise with strength {scale.item()}.')
        else:
            self.generator = None
            
        if local_diff_privacy["input_noise"] > 0.0:
            loc = torch.as_tensor(0.0, **setup)
            scale = torch.as_tensor(local_diff_privacy["input_noise"], **setup)
            if local_diff_privacy["distribution"] == "gaussian":
                self.generator_input = torch.distributions.normal.Normal(loc=loc, scale=scale)
            elif local_diff_privacy["distribution"] == "laplacian":
                self.generator_input = torch.distributions.laplace.Laplace(loc=loc, scale=scale)
            else:
                raise ValueError(f'Invalid distribution {local_diff_privacy["distribution"]} given.')
            self.defense_repr.append(
                f'Defense: Local {local_diff_privacy["distribution"]} input noise with strength {scale.item()}.'
            )
        else:
            self.generator_input = None
        self.clip_value = local_diff_privacy.get("per_example_clipping", 0.0)
        if self.clip_value > 0:
            self.defense_repr.append(f"Defense: Gradient clipping to maximum of {self.clip_value}.")
            
            
    def applyNoise(self, gradient):
        """Apply differential privacy component gradient noise."""
        if self.generator is not None:
            for grad in gradient:
                grad += self.generator.sample(grad.shape)
        return gradient
        
    def clampGradient(self, model, data, batch_size):
        def _compute_batch_gradient(data):
            data[self.data_key] = (
                data[self.data_key] + self.generator_input.sample(data[self.data_key].shape)
                if self.generator_input is not None
                else data[self.data_key]
            )
            input_data = data["inputs"].unsqueeze(1) # Change to inputs, add dimension for channel MH_2025
            outputs = model(input_data)
            loss = self.loss(outputs, data["labels"])
            return torch.autograd.grad(loss, model.parameters())

        if self.clip_value > 0:  # Compute per-example gradients and clip them in this case
            shared_grads = [torch.zeros_like(p) for p in model.parameters()]
            for data_idx in range(batch_size):
                data_point = {key: val[data_idx : data_idx + 1] for key, val in data.items()}
                per_example_grads = _compute_batch_gradient(data_point)
                self._clip_list_of_grad_(per_example_grads)
                torch._foreach_add_(shared_grads, per_example_grads)
            torch._foreach_div_(shared_grads, batch_size)
        else:
            # Compute the forward pass
            shared_grads = _compute_batch_gradient(data)
            
    def _clip_list_of_grad_(self, gradient):
        """Apply differential privacy component per-example clipping."""
        grad_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in gradient]), 2)
        if grad_norm > self.clip_value:
            [g.mul_(self.clip_value / (grad_norm + 1e-6)) for g in gradient]
        return gradient