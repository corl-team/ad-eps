import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


class CNNExtractor(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        n_filters: int = 64,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, n_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = (
                self.cnn(torch.as_tensor(observation_space.sample()[None]).float())
                .flatten()
                .shape[0]
            )

        self.linear = nn.Linear(n_flatten, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
