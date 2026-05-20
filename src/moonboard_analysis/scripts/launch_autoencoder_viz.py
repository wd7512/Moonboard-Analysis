"""Gradio app for autoencoder latent space exploration.

Usage:
    uv run python scripts/launch_autoencoder_viz.py

Loads a trained autoencoder and training data, then launches an interactive
web interface for manipulating latent vectors and visualizing reconstructions
on a Moonboard grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import torch

from moonboard_analysis.config import AutoencoderConfig
from moonboard_analysis.data.grid_mapping import GridMapper
from moonboard_analysis.models.autoencoder import Autoencoder
from moonboard_analysis.utils.device import get_device
from moonboard_analysis.visualization.renderer import GridRenderer

matplotlib.use("Agg")

GRADE_MAP = {
    0: "6B+",
    1: "6C",
    2: "6C+",
    3: "7A",
    4: "7A+",
    5: "7B",
    6: "7B+",
    7: "7C",
    8: "7C+",
    9: "8A",
    10: "8A+",
    11: "8B",
    12: "8B+",
}


def _decode_grade(grade_idx: float) -> str:
    """Convert grade index to grade string."""
    idx = int(round(grade_idx))
    return GRADE_MAP.get(idx, f"Unknown ({idx})")


def _load_model(model_path: str, device: torch.device) -> Autoencoder:
    """Load trained autoencoder from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    config = checkpoint.get("config", {})
    input_dim = config.get("input_dim", AutoencoderConfig.input_dim)
    bottleneck_dim = config.get("bottleneck_dim", AutoencoderConfig.bottleneck_dim)
    bounded = config.get("bounded", False)

    model = Autoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim, bounded=bounded)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _load_data(data_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load training data and return grades, features, and repeats."""
    data = np.load(data_path, allow_pickle=True)
    grades = np.array([row[0] for row in data], dtype=float)
    features = np.stack([row[1] for row in data]).astype(np.float32)

    repeats_path = Path(data_path).parent / "2016Repeats164.npy"
    if repeats_path.exists():
        repeats = np.load(repeats_path)
    else:
        repeats = np.zeros(len(grades), dtype=int)
    return grades, features, repeats


def _get_top_routes_per_grade(
    grades: np.ndarray,
    repeats: np.ndarray,
    n_per_grade: int = 4,
) -> np.ndarray:
    """Return indices of the top N most popular routes per grade."""
    selected: list[int] = []
    for grade_idx in range(len(GRADE_MAP)):
        mask = grades == grade_idx
        grade_indices = np.where(mask)[0]
        grade_repeats = repeats[mask]
        top_local = np.argsort(grade_repeats)[-n_per_grade:][::-1]
        selected.extend(grade_indices[top_local].tolist())
    return np.array(selected, dtype=int)


def _build_route_labels(
    grades: np.ndarray,
    repeats: np.ndarray,
    route_indices: np.ndarray,
) -> list[str]:
    """Build dropdown labels for filtered routes."""
    labels = []
    for idx in route_indices:
        grade_idx = int(grades[idx])
        grade_str = _decode_grade(grade_idx)
        rep = int(repeats[idx])
        labels.append(f"Route #{idx} — Grade: {grade_str} — {rep:,} repeats")
    return labels


def _compute_latent_ranges(
    model: Autoencoder,
    features: np.ndarray,
    device: torch.device,
    bottleneck_dim: int,
) -> list[tuple[float, float]]:
    """Compute p5/p95 ranges for each latent dimension."""
    tensor = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        encoded = model.encode(tensor).cpu().numpy()

    ranges = []
    for dim in range(bottleneck_dim):
        p5 = float(np.percentile(encoded[:, dim], 5))
        p95 = float(np.percentile(encoded[:, dim], 95))
        margin = (p95 - p5) * 0.1
        ranges.append((p5 - margin, p95 + margin))
    return ranges


def create_app(
    model: Autoencoder,
    grades: np.ndarray,
    features: np.ndarray,
    latent_ranges: list[tuple[float, float]],
    mapper: GridMapper,
    renderer: GridRenderer,
    device: torch.device,
    bottleneck_dim: int = 8,
    route_indices: np.ndarray | None = None,
    repeats: np.ndarray | None = None,
    bounded: bool = False,
) -> gr.Blocks:
    """Create the Gradio Blocks interface.

    Args:
        model: Trained autoencoder.
        grades: Grade indices for all routes.
        features: 164-dim feature vectors for all routes.
        latent_ranges: (min, max) for each latent dimension.
        mapper: GridMapper for coordinate conversion.
        renderer: GridRenderer for visualization.
        device: Torch device for model inference.
        bottleneck_dim: Dimension of the latent space.
        route_indices: Indices of routes to show in dropdown.
        repeats: Repeat counts for each route.
        bounded: Whether the model uses tanh-bounded latent space.

    Returns:
        Gradio Blocks app.
    """
    if route_indices is None:
        route_indices = np.arange(len(grades))
    if repeats is None:
        repeats = np.zeros(len(grades), dtype=int)

    route_labels = _build_route_labels(grades, repeats, route_indices)

    slider_ranges = [(-1.0, 1.0)] * bottleneck_dim if bounded else latent_ranges

    with gr.Blocks(title="Moonboard Autoencoder Explorer") as app:
        gr.Markdown("# Moonboard Autoencoder — Latent Space Explorer")
        gr.Markdown(
            "Select a route, then manipulate its 8-dimensional latent "
            "representation to see how the reconstruction changes."
        )

        with gr.Row():
            with gr.Column(scale=1):
                route_dropdown = gr.Dropdown(
                    choices=route_labels,
                    value=route_labels[0],
                    label="Route",
                )
                threshold_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    label="Binarization Threshold",
                )

                gr.Markdown("### Latent Dimensions")
                sliders: list = []
                for dim in range(bottleneck_dim):
                    low, high = slider_ranges[dim]
                    slider = gr.Slider(
                        minimum=low,
                        maximum=high,
                        value=0.0,
                        step=0.01,
                        label=f"Dimension {dim}",
                    )
                    sliders.append(slider)

                with gr.Row():
                    reset_btn = gr.Button("Reset to Encoded")
                    random_btn = gr.Button("Randomize")

            with gr.Column(scale=2):
                mse_text = gr.Textbox(label="Reconstruction MSE", interactive=False)
                comparison_plot = gr.Plot(label="Original vs Reconstructed")

        def _get_route_index(label: str) -> int:
            return int(label.split("#")[1].split(" ")[0])

        def update_from_route(route_label: str) -> list:
            idx = _get_route_index(route_label)
            feature = features[idx : idx + 1]
            tensor = torch.tensor(feature, dtype=torch.float32).to(device)
            with torch.no_grad():
                encoded = model.encode(tensor).cpu().numpy()[0]
            return [float(v) for v in encoded]

        def update_visualization(
            route_label: str,
            threshold: float,
            *latent_values: float,
        ) -> tuple:
            idx = _get_route_index(route_label)
            original_vec = features[idx]
            threshold = float(threshold)

            latent = np.array(
                [0.0 if v is None else float(v) for v in latent_values],
                dtype=np.float32,
            ).reshape(1, -1)
            latent_tensor = torch.tensor(latent, dtype=torch.float32).to(device)
            with torch.no_grad():
                reconstructed = model.decode(latent_tensor).cpu().numpy()[0]

            mse = float(np.mean((original_vec - reconstructed) ** 2))

            original_grid = mapper.vector_to_grid(original_vec)
            recon_grid = mapper.vector_to_grid(reconstructed)

            fig = renderer.render_comparison(
                original_grid,
                recon_grid,
                threshold=threshold,
            )

            return fig, f"{mse:.6f}"

        def reset_sliders(route_label: str) -> list:
            values = update_from_route(route_label)
            return values

        def randomize_sliders() -> list:
            values = []
            for dim in range(bottleneck_dim):
                low, high = slider_ranges[dim]
                values.append(float(np.random.uniform(low, high)))
            return values

        for slider in sliders:
            deps = [route_dropdown, threshold_slider] + sliders
            slider.change(
                fn=update_visualization,
                inputs=deps,
                outputs=[comparison_plot, mse_text],
            )

        route_dropdown.change(
            fn=reset_sliders,
            inputs=[route_dropdown],
            outputs=sliders,
        ).then(
            fn=update_visualization,
            inputs=[route_dropdown, threshold_slider] + sliders,
            outputs=[comparison_plot, mse_text],
        )

        threshold_slider.change(
            fn=update_visualization,
            inputs=[route_dropdown, threshold_slider] + sliders,
            outputs=[comparison_plot, mse_text],
        )

        reset_btn.click(
            fn=reset_sliders,
            inputs=[route_dropdown],
            outputs=sliders,
        ).then(
            fn=update_visualization,
            inputs=[route_dropdown, threshold_slider] + sliders,
            outputs=[comparison_plot, mse_text],
        )

        random_btn.click(
            fn=randomize_sliders,
            inputs=[],
            outputs=sliders,
        ).then(
            fn=update_visualization,
            inputs=[route_dropdown, threshold_slider] + sliders,
            outputs=[comparison_plot, mse_text],
        )

        app.load(
            fn=reset_sliders,
            inputs=[route_dropdown],
            outputs=sliders,
        ).then(
            fn=update_visualization,
            inputs=[route_dropdown, threshold_slider] + sliders,
            outputs=[comparison_plot, mse_text],
        )

    return app


def main() -> None:
    """Parse arguments and launch the Gradio app."""
    parser = argparse.ArgumentParser(description="Autoencoder latent space visualizer")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Autoencoder_Moonboard.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="archive/Legacy/2016TrainingData164.npy",
        help="Path to training data .npy file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to serve the app on",
    )
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"Error: Model not found at '{args.model_path}'")
        return

    if not Path(args.data_path).exists():
        print(f"Error: Data not found at '{args.data_path}'")
        return

    device = get_device()
    print(f"Loading model from {args.model_path} on {device}")
    model = _load_model(args.model_path, device)

    print(f"Loading data from {args.data_path}")
    grades, features, repeats = _load_data(args.data_path)
    print(f"Loaded {len(features)} routes with {features.shape[1]} features")

    route_indices = _get_top_routes_per_grade(grades, repeats, n_per_grade=4)
    print(f"Showing top {4} routes per grade: {len(route_indices)} total")

    print("Computing latent space ranges...")
    bottleneck_dim = model.bottleneck_dim
    bounded = model.bounded
    latent_ranges = _compute_latent_ranges(model, features, device, bottleneck_dim)

    mapper = GridMapper()
    renderer = GridRenderer(mapper)

    print("Building Gradio interface...")
    app = create_app(
        model, grades, features, latent_ranges, mapper, renderer, device,
        bottleneck_dim, route_indices, repeats, bounded,
    )

    print(f"Launching app on http://localhost:{args.port}")
    app.launch(server_port=args.port)


if __name__ == "__main__":
    main()
