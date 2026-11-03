"""Identity-compatible residual graph expansion for a cached [BASE_MODEL] region."""
from __future__ import annotations

import torch
from torch import nn


class IdentityResidualConnector(nn.Module):
    """Low-rank residual branch initialized to contribute exactly zero."""

    def __init__(self, hidden_size: int = 3072, rank: int = 8):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(rank, hidden_size, bias=False, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(
            torch.nn.functional.linear(hidden.float(), self.down.weight), self.up.weight
        ).to(hidden.dtype)


class CachedActivationLayer(nn.Module):
    """Return a verified frozen prefix activation for one fixed input record."""

    def __init__(self, activation: torch.Tensor):
        super().__init__()
        self.register_buffer("activation", activation.detach().clone(), persistent=False)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        return self.activation


class ConnectedCachedActivationLayer(CachedActivationLayer):
    """Insert a trainable residual connector at the layer 26 -> 27 boundary."""

    def __init__(self, activation: torch.Tensor, connector: IdentityResidualConnector):
        super().__init__(activation)
        self.connector = connector

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        cached = self.activation
        return cached + self.connector(cached)


class ConnectedDecoderLayer(nn.Module):
    """Run the genuine decoder layer, then add a trainable hidden-state branch."""

    def __init__(self, decoder_layer: nn.Module, connector: IdentityResidualConnector):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.connector = connector

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        layer_output = self.decoder_layer(hidden_states, *args, **kwargs)
        if isinstance(layer_output, (tuple, list)):
            raise TypeError("ConnectedDecoderLayer expects a tensor-returning Transformers LlamaDecoderLayer")
        return layer_output + self.connector(layer_output)


def install_cached_prefix(model, activations: list[torch.Tensor], connector: IdentityResidualConnector | None = None):
    """Replace the first 27 decoder blocks with captured outputs for exact inputs."""
    if len(model.model.layers) != 28 or len(activations) != 27:
        raise ValueError("Expected [BASE_MODEL] Llama 3B with 28 layers and 27 captured prefix activations")
    original = list(model.model.layers)
    for index, activation in enumerate(activations):
        if index == 26 and connector is not None:
            model.model.layers[index] = ConnectedCachedActivationLayer(activation, connector)
        else:
            model.model.layers[index] = CachedActivationLayer(activation)
    return original


def install_cached_prefix_live_connector(model, activations: list[torch.Tensor], connector: IdentityResidualConnector):
    """Cache layers 0–25; run real layer 26 with an attached connector and live layer 27."""
    if len(model.model.layers) != 28 or len(activations) != 27:
        raise ValueError("Expected [BASE_MODEL] Llama 3B with 28 layers and 27 captured prefix activations")
    original = list(model.model.layers)
    for index in range(26):
        model.model.layers[index] = CachedActivationLayer(activations[index])
    model.model.layers[26] = ConnectedDecoderLayer(original[26], connector)
    return original


def attach_live_connector(model, connector: IdentityResidualConnector):
    """Attach the validated connector to the real layer 26 for uncached inputs."""
    if len(model.model.layers) != 28:
        raise ValueError("Expected a 28-layer [BASE_MODEL] Llama decoder")
    original = model.model.layers[26]
    model.model.layers[26] = ConnectedDecoderLayer(original, connector)
    return original


def remove_live_connector(model, original_layer: nn.Module) -> None:
    if len(model.model.layers) != 28:
        raise ValueError("Expected a 28-layer [BASE_MODEL] Llama decoder")
    model.model.layers[26] = original_layer


def restore_original_layers(model, original_layers: list[nn.Module]) -> None:
    if len(original_layers) != 28:
        raise ValueError("Invalid original [BASE_MODEL] decoder layer snapshot")
    model.model.layers = nn.ModuleList(original_layers)
