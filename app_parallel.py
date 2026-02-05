import dash
import re as regex
import os
from dash import dcc, html, Input, Output, State
import plotly.express as px

from pathlib import Path
import math

import numpy as np
import pandas as pd
from sympy import re
import umap

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# from plot_embedding import parse_stats_file

STATS_DIR = "./summary_stats_positive_seventy_parallel"
TOP_K_FEATURES = 800
FEATURE_GROUPS = []

STAT_GROUPS = {
    "low_band_moments": ["Low-band skewness and kurtosis"],
    "high_band_variance": ["High-band variance"],
    "reconstruction_stats": ["Skewness, kurtosis, variance, mean, maximum and minimum of the reconstructed texture"],
    "lowband_autocorr": ["Auto-correlations of the low-frequency bands at each scale"],
    "oriented_autocorr": ["Auto-correlations of the modulus of each oriented sub-band"],
    "same_scale_crosscorr": ["Pairwise cross-correlations of the modulus of all oriented sub-bands at the same scale (size QxQ)"],
    "cross_scale_mag": ["Cross-correlations of the modulus of all oriented sub-bands with all oriented sub-bands at the coarser scale (size QxQ)"],
    "cross_scale_phase": ["Cross-correlations of the real part of each oriented sub-band with both the real and imaginary part of all phase-doubled oriented sub-bands at the next coarser scale (Q matrices of size 1x2Q, which are computed and stored in a matrix of size Qx2Q)"],
}

STAT_COLORS = {
    "low_band_moments": "#1f77b4",        # blue
    "high_band_variance": "#ff7f0e",      # orange
    "reconstruction_stats": "#2ca02c",    # green
    "lowband_autocorr": "#d62728",        # red
    "oriented_autocorr": "#9467bd",       # purple
    "same_scale_crosscorr": "#8c564b",    # brown
    "cross_scale_mag": "#e377c2",         # pink
    "cross_scale_phase": "#7f7f7f",       # gray
}

def build_image_index(img_dir):
    """
    Maps realized correlation -> image basename
    scatter_bw_70_0.58_p.png → 0.58
    """
    index = []

    for p in Path(img_dir).glob("parallel_bw_70_*.png"):
        stem = p.stem  # parallel_bw_70_0.58_p
        parts = stem.split("_")
        try:
            corr = float(parts[-2])
            sign = parts[-1]
            index.append((corr, sign, stem))
        except ValueError:
            continue

    return index

def closest_image(target_corr, image_index):
    corr, sign, stem = min(
        image_index,
        key=lambda x: abs(x[0] - target_corr)
    )
    return corr, sign, stem

def target_corr_from_stats(fname):
    """
    stats_corr_scatter_bw_70_0.000_z.txt → 0.000
    """
    parts = fname.replace(".txt", "").split("_")
    return float(parts[-2])

def corr_from_filename(fname):
    """
    Extract correlation from:
    statistics_scatter_bw_8_-0.50_n.txt  →  -0.50
    """
    base = os.path.basename(fname)
    stem = base.replace("statistics_", "").replace(".txt", "")
    parts = stem.split("_")

    try:
        corr = float(parts[-2])
    except (IndexError, ValueError):
        raise ValueError(f"Could not parse correlation from filename: {fname}")

    return corr

def parse_stats_file(path):
    with open(path, "r") as f:
        lines = f.readlines()

    stats = {}
    current_key = None
    current_values = []

    def flush():
        if current_key is not None and current_values:
            stats[current_key] = [
                float(x)
                for line in current_values
                for x in line.replace(",", "").split()
            ]

    for line in lines:
        line = line.strip()

        if not line:
            flush()
            current_key = None
            current_values = []
            continue

        if any(
            line.startswith(prefix)
            for group in STAT_GROUPS.values()
            for prefix in group
        ):
            flush()
            current_key = line
            current_values = []
        else:
            current_values.append(line)

    flush()
    return stats

def build_feature_template(example_stats, enabled_groups):
    template = {}
    feature_names = []

    for group in enabled_groups:
        for stat_name in STAT_GROUPS[group]:
            if stat_name not in example_stats:
                continue

            values = example_stats[stat_name]
            idxs = list(range(len(values)))

            template[stat_name] = idxs

            for i in idxs:
                feature_names.append(f"{stat_name}[{i}]")

    return template, feature_names

def vectorize_stats(stats, template, enabled_groups):
    vec = []

    for group in enabled_groups:
        for stat_name in STAT_GROUPS[group]:
            if stat_name not in template:
                continue

            values = stats.get(stat_name, [])
            idxs = template[stat_name]

            for i in idxs:
                vec.append(values[i] if i < len(values) else 0.0)

    return np.array(vec, dtype=float)

def build_dataset_matrix(
    stats_dir,
    template,
    feature_names,
    enabled_groups=None,
):
    X = []
    meta = []

    for fname in sorted(os.listdir(stats_dir)):
        if not fname.endswith(".txt"):
            continue

        path = os.path.join(stats_dir, fname)

        # Parse full summary stats (single vector per image)
        stats = parse_stats_file(path)
        vec = vectorize_stats(stats, template, enabled_groups)
        X.append(vec)

        # Recover image base name
        # statistics_scatter_bw_70_-0.35_z.txt → scatter_bw_70_-0.35_z
        image_name = (fname.replace("statistics_", "").replace(".txt", ""))
        corr = corr_from_filename(image_name)

        IMAGE_INDEX = build_image_index("assets/original_positive_seventy_parallel")

        target_corr = target_corr_from_stats(fname)

        actual_corr, sign, image_name = closest_image(
            target_corr,
            IMAGE_INDEX
        )

        meta.append({
            "image": image_name,
            "corr": actual_corr,
            "orig_img": f"/assets/original_positive_seventy_parallel/{image_name}.png",
            "out_img": f"/assets/output_positive_seventy_parallel/{image_name}_output.png",
        })

    X = np.stack(X, axis=0)   # (N, F)

    return X, meta

# -----------------------------
# Load full dataset once
# -----------------------------
# -----------------------------
# Build feature template from ONE example file
# -----------------------------
example_fname = next(
    f for f in sorted(os.listdir(STATS_DIR)) if f.endswith(".txt")
)
example_path = os.path.join(STATS_DIR, example_fname)
example_stats = parse_stats_file(example_path)

TEMPLATE, FEATURE_NAMES = build_feature_template(
    example_stats,
    enabled_groups=list(STAT_GROUPS.keys())
)

FEATURE_GROUPS = []

# build mapping
FEATURE_TO_GROUP = {}
for group, keys in STAT_GROUPS.items():
    for key in keys:
        for f in FEATURE_NAMES:
            if f.startswith(key):
                FEATURE_TO_GROUP[f] = group

X, META = build_dataset_matrix(
    STATS_DIR, TEMPLATE, FEATURE_NAMES, list(STAT_GROUPS.keys())
)

ALL_IMAGES = sorted({m["image"] for m in META})

N, F = X.shape
assert F == len(FEATURE_NAMES)

X_scaled = StandardScaler().fit_transform(X)

n_pca = min(30, N - 1, F)
pca = PCA(n_components=n_pca, svd_solver="full", whiten=True, random_state=0)
X_pca = pca.fit_transform(X_scaled)

components = pca.components_        # (K, F)
explained = pca.explained_variance_ratio_

feature_importance = np.sum(
    np.abs(components) * explained[:, None], axis=0
)

df_features = pd.DataFrame({
    "feature": FEATURE_NAMES,
    "importance": feature_importance,
})

# Map features → statistic groups
df_features["group"] = df_features["feature"].map(FEATURE_TO_GROUP)

# Safety fallback
df_features["group"] = df_features["group"].fillna("unknown")

# Assign colors per group (matplotlib-safe hex)
df_features["color"] = df_features["group"].map(STAT_COLORS).fillna("#333333")

# Sort by importance (elbow plot order)
df_features = (
    df_features
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

# Global rank (0 … 942)
df_features["rank"] = np.arange(len(df_features))

# ---- Aggregate importance per statistic group ----
df_group = (
    df_features
    .groupby("group")
    .agg(
        group_importance=("importance", "sum"),   # total contribution
        mean_rank=("rank", "mean"),                # x-position on elbow
        median_rank=("rank", "median"),
        n_features=("feature", "size"),
    )
    .reset_index()
)

# Sort groups by importance (descending)
df_group = df_group.sort_values(
    "group_importance", ascending=False
).reset_index(drop=True)


# Compute within-group index and group sizes
df_features["group_index"] = (
    df_features
    .groupby("group")
    .cumcount()
)

# group_sizes = df_features.groupby("group").size().to_dict()
# df_features["group_size"] = df_features["group"].map(group_sizes)

df_features["group_size"] = (
    df_features
    .groupby("group")["feature"]
    .transform("size")
)

df_features["group_running_mean"] = np.nan

# Running mean importance per statistic group
for group in STAT_GROUPS.keys():
    mask = df_features["group"] == group
    vals = df_features.loc[mask, "rank"].values
    if len(vals) > 0:
        df_features.loc[mask, "group_running_mean"] = (
            np.cumsum(vals) / (np.arange(len(vals)) + 1)
        )

# Sanity check: number of features per group
print(df_features.groupby("group").size())

# Plot elbow
elbow_fig = px.scatter(
    df_features,
    x="rank",
    y="importance",
    color="group",
    color_discrete_map=STAT_COLORS,
    hover_data={
        "group": True,
        "group_index": True,
        "group_size": True,
        "importance": ":.4e",
        "rank": True,
    },
)

elbow_fig.update_traces(
    marker=dict(size=6, opacity=0.8),
    hovertemplate=(
        "<b>Feature rank:</b> %{x}<br>"
        "<b>Group:</b> %{customdata[0]}<br>"
        "<b>Position in group:</b> %{customdata[1]} / %{customdata[2]}<br>"
        "<b>Importance:</b> %{customdata[3]}<extra></extra>"
    )
)

elbow_fig.update_layout(
    title="PCA Feature Importance Elbow (colored by statistic group)",
    xaxis_title="Feature rank (sorted by importance)",
    yaxis_title="PCA feature importance",
    height=400,
    legend_title="Statistic group",
)

print(df_group[["group", "group_importance", "n_features"]])

TOP_FEATURES = df_features.head(TOP_K_FEATURES)
TOP_FEATURE_INDICES = [
    FEATURE_NAMES.index(f) for f in TOP_FEATURES["feature"]
]

X_top = X[:, TOP_FEATURE_INDICES]   # (N, K)

# Scale
# X_top_scaled = StandardScaler().fit_transform(X_top)

# Use all features directly, skip PCA
# X_final = X_top_scaled

X_final = StandardScaler().fit_transform(X)

print("Using features:", X_final.shape[-1])  # should be 1200
print("Original features:", X.shape[-1])     # 1200

# print("Using features:", X_pca_top.shape[-1])  # should be 100
# print("Original features:", X.shape[-1])   # 943

# DASH
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id="elbow-plot", figure=elbow_fig),
    # LEFT: controls
    html.Div([
        html.H3("PS Statistic Explorer"),
        html.Label("Top K features:"),
        dcc.Input(
            id="topk-input",
            type="number",
            value=TOP_K_FEATURES,  # default 800
            min=1,
            max=len(FEATURE_NAMES),
            step=1,
        ),
        html.Br(),
        html.Label("Select images"),
        dcc.Dropdown(
            ALL_IMAGES,
            ALL_IMAGES,
            multi=True,
            id="image-select"
        ),
        html.Label("Select statistic groups"),
        dcc.Checklist(
            options=[{"label": k, "value": k} for k in STAT_GROUPS],
            value=list(STAT_GROUPS.keys()),
            id="stat-select"
        ),
        html.Button("Recompute embedding", id="recompute"),
    ], style={"width": "20%", "float": "left"}),

    # CENTER: UMAP
    html.Div([
        dcc.Graph(id="umap-plot")
    ], style={"width": "55%", "float": "left"}),

    # RIGHT: image preview
    html.Div([
        html.H4("Hovered texture"),
        html.Img(id="orig-img", style={"width": "256px", "border": "1px solid black"}),
        html.Br(),
        html.Img(id="out-img", style={"width": "256px", "border": "1px solid black"}),
    ], style={"width": "25%", "float": "right"}),

])

@app.callback(
    Output("umap-plot", "figure"),
    Input("image-select", "value"),
    Input("stat-select", "value"),
    Input("topk-input", "value"),
    State("recompute", "n_clicks"),
)

def update_embedding(selected_images, enabled_stats, topk_value, _):

    mask = [m["image"] in selected_images for m in META]
    meta = [m for m, keep in zip(META, mask) if keep]

    topk_value = min(max(1, topk_value), X_final.shape[1])

    top_features = df_features.head(topk_value)
    top_indices = [FEATURE_NAMES.index(f) for f in top_features["feature"]]

    # X = StandardScaler().fit_transform(X_pca_top[mask])
    X = X_final[mask][:, top_indices]  # Already scaled

    print(f"Computing UMAP for {X.shape[0]} images using top {topk_value} features...")
    print(f"Computing UMAP for {X.shape[0]} images with {len(enabled_stats)} statistic groups...")

    Xu = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=0, n_epochs=500, init="spectral").fit_transform(X)
    df = pd.DataFrame({
        "x": Xu[:, 0],
        "y": Xu[:, 1],
        "corr": [m["corr"] for m in meta],
        "image": [m["image"] for m in meta],
        "orig": [m["orig_img"] for m in meta],
        "out": [m["out_img"] for m in meta],
    })

    # Ensure corr is numeric and finite
    df["corr"] = pd.to_numeric(df["corr"], errors="coerce").fillna(0.0)

    abs_corr = np.abs(df["corr"].values)
    abs_corr = np.clip(abs_corr, 0, 1)
    size_vals = 1 + 10 * (abs_corr ** 2.5)

    fig = px.scatter(
        df,
        x="x",
        y="y",
        size=size_vals,
        size_max=20,
        hover_data={"image": True, "corr": True},
    )

    fig.update_traces(
            customdata=df[["image", "orig", "out", "corr"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>corr=%{customdata[3]:.2f}<extra></extra>",
            marker=dict(
                color="black",
                opacity=0.6
            )
        )

    # for _, row in df.iterrows():
    #     fig.add_annotation(
    #         x=row["x"],
    #         y=row["y"],
    #         text=f"{row['corr']:.2f}",
    #         showarrow=False,
    #         font=dict(size=10, color="black"),
    #         yshift=-15
    #     )

    return fig

@app.callback(
    Output("orig-img", "src"),
    Output("out-img", "src"),
    Input("umap-plot", "hoverData"),
)
def update_images(hoverData):
    if hoverData is None:
        return dash.no_update, dash.no_update

    point = hoverData["points"][0]
    image_name, orig, out, corr = point["customdata"]

    return orig, out

if __name__ == "__main__":
    app.run(debug=True)