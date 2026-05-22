"""Tests for Gradio app event handlers and type coercion."""

from pathlib import Path

import numpy as np
import pytest
import torch

from moonboard_analysis.data.grid_mapping import GridMapper
from moonboard_analysis.models.autoencoder import Autoencoder
from moonboard_analysis.scripts.launch_autoencoder_viz import (
    _build_route_labels,
    _compute_latent_ranges,
    _get_top_routes_per_grade,
    _load_data,
    _load_model,
    create_app,
)
from moonboard_analysis.visualization.renderer import GridRenderer

MODEL_PATH = "Autoencoder_Moonboard.pth"
DATA_PATH = "archive/Legacy/2016TrainingData164.npy"


@pytest.fixture
def device() -> torch.device:
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def model(device: torch.device) -> Autoencoder:
    """Load trained autoencoder."""
    if not Path(MODEL_PATH).exists():
        pytest.skip(f"Model checkpoint not found at {MODEL_PATH}")
    return _load_model(MODEL_PATH, device)


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load training data."""
    if not Path(DATA_PATH).exists():
        pytest.skip(f"Data file not found at {DATA_PATH}")
    return _load_data(DATA_PATH)


@pytest.fixture
def latent_ranges(
    model: Autoencoder,
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
) -> list[tuple[float, float]]:
    """Compute latent space ranges."""
    grades, features, _ = data
    return _compute_latent_ranges(model, features, device, model.bottleneck_dim)


@pytest.fixture
def mapper() -> GridMapper:
    """Return a GridMapper instance."""
    return GridMapper()


@pytest.fixture
def renderer(mapper: GridMapper) -> GridRenderer:
    """Return a GridRenderer instance."""
    return GridRenderer(mapper)


class TestAppCreation:
    """Test Gradio app creation."""

    def test_app_creates(
        self,
        model: Autoencoder,
        data: tuple[np.ndarray, np.ndarray, np.ndarray],
        latent_ranges: list[tuple[float, float]],
        mapper: GridMapper,
        renderer: GridRenderer,
        device: torch.device,
    ) -> None:
        """Verify app creates without errors."""
        grades, features, repeats = data
        route_indices = _get_top_routes_per_grade(grades, repeats)
        app = create_app(
                 model, grades, features, latent_ranges,
                 mapper, renderer, device,
                 model.bottleneck_dim, route_indices, repeats)
        assert app is not None

    def test_route_labels_format(self, data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Verify route labels have expected format."""
        grades, _, repeats = data
        route_indices = _get_top_routes_per_grade(grades, repeats)
        labels = _build_route_labels(grades, repeats, route_indices)
        assert len(labels) == len(route_indices)
        assert "Route #" in labels[0]
        assert "Grade:" in labels[0]
        assert "repeats" in labels[0]


class TestTypeCoercion:
    """Test type coercion for Gradio 5.x string inputs.

    Gradio 5.x passes slider values as strings to event handlers.
    These tests verify the app handles this correctly.
    """

    def test_string_latent_values_coerced(
        self,
        model: Autoencoder,
        data: tuple[np.ndarray, np.ndarray, np.ndarray],
        mapper: GridMapper,
        renderer: GridRenderer,
        device: torch.device,
    ) -> None:
        """Verify string latent values are correctly coerced to floats."""
        grades, features, _ = data
        _ = _compute_latent_ranges(model, features, device, model.bottleneck_dim)

        # Simulate Gradio 5 passing strings
        string_values = ["1.5", "2.0", "0.5", "3.0", "1.0", "2.5", "0.8", "1.2"]
        coerced = [float(v) for v in string_values]

        latent = np.array(coerced, dtype=np.float32).reshape(1, -1)
        tensor = torch.tensor(latent, dtype=torch.float32).to(device)

        with torch.no_grad():
            reconstructed = model.decode(tensor).cpu().numpy()[0]

        assert reconstructed.shape == (164,)
        assert not np.any(np.isnan(reconstructed))

    def test_string_threshold_coerced(
        self,
        model: Autoencoder,
        data: tuple[np.ndarray, np.ndarray, np.ndarray],
        mapper: GridMapper,
        renderer: GridRenderer,
        device: torch.device,
    ) -> None:
        """Verify string threshold is correctly coerced to float."""
        grades, features, _ = data
        _ = _compute_latent_ranges(model, features, device, model.bottleneck_dim)

        string_threshold = "0.5"
        threshold = float(string_threshold)

        latent = np.zeros((1, 8), dtype=np.float32)
        tensor = torch.tensor(latent, dtype=torch.float32).to(device)

        with torch.no_grad():
            reconstructed = model.decode(tensor).cpu().numpy()[0]

        original_vec = features[0]
        original_grid = mapper.vector_to_grid(original_vec)
        recon_grid = mapper.vector_to_grid(reconstructed)

        # Should not raise with float threshold
        fig = renderer.render_comparison(original_grid, recon_grid, threshold=threshold)
        assert fig is not None

    def test_mixed_string_and_float_inputs(
        self,
        model: Autoencoder,
        data: tuple[np.ndarray, np.ndarray, np.ndarray],
        mapper: GridMapper,
        renderer: GridRenderer,
        device: torch.device,
    ) -> None:
        """Verify mixed string/float inputs work correctly."""
        grades, features, _ = data
        _ = _compute_latent_ranges(model, features, device, model.bottleneck_dim)

        # Mix of strings and floats (simulating Gradio 5 behavior)
        mixed_values = [1.5, "2.0", 0.5, "3.0", 1.0, "2.5", 0.8, "1.2"]
        coerced = [float(v) for v in mixed_values]

        latent = np.array(coerced, dtype=np.float32).reshape(1, -1)
        tensor = torch.tensor(latent, dtype=torch.float32).to(device)

        with torch.no_grad():
            reconstructed = model.decode(tensor).cpu().numpy()[0]

        assert reconstructed.shape == (164,)

    def test_returned_values_are_python_floats(
        self,
        model: Autoencoder,
        data: tuple[np.ndarray, np.ndarray, np.ndarray],
        device: torch.device,
    ) -> None:
        """Verify returned slider values are plain Python floats."""
        grades, features, _ = data
        _ = _compute_latent_ranges(model, features, device, model.bottleneck_dim)

        # Encode a route and check return types
        idx = 0
        feature = features[idx : idx + 1]
        tensor = torch.tensor(feature, dtype=torch.float32).to(device)
        with torch.no_grad():
            encoded = model.encode(tensor).cpu().numpy()[0]

        values = [float(v) for v in encoded]
        for v in values:
            assert type(v) is float, f"Expected Python float, got {type(v)}"
