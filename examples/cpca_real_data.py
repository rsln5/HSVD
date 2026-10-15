"""Real mouse-protein PCA/cPCA, using the general HSVD algorithm.

Run ``python3 -m examples.cpca_real_data`` to regenerate all artifacts offline.
Run ``python3 -m examples.cpca_real_data --check`` to replay and verify them.
The CC BY 4.0 UCI CSV is cached with provenance in results/applications/cpca_real.
The author's notebook supplies the split, zero imputation, scaling and alpha=2;
genotype labels are used only to color the target observations, never to fit.
This reproduces that explicit notebook configuration, not every setting of the
published figure. The 15 records per mouse are technical measurements.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np

from hsvd import hsvd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/applications/cpca_real"
DATA_SHA256 = "86dc3961c391df56c98d86757f779ae04e19229f12d9835ca084d01de22a10e5"
DATA_URL = "https://archive.ics.uci.edu/static/public/342/data.csv"
ALPHA = 2.0
ARTIFACTS = ("data.csv", "inputs.npz", "computed.npz", "projection.csv", "pca_vs_cpca.png",
             "pca_vs_cpca.pdf", "PROVENANCE.md")
SOURCES = ("examples/cpca_real_data.py", "tests/test_cpca_real_data.py",
           "hsvd/core.py", "hsvd/_numeric.py", "hsvd/_types.py", "hsvd/__init__.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_dataset(path):
    """Read the pinned UCI table without requiring pandas or an Excel reader."""
    if sha256(path) != DATA_SHA256:
        raise ValueError("UCI CSV checksum differs from the documented cached input")
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        features = reader.fieldnames[1:78]
        rows = list(reader)
    if len(features) != 77 or len(rows) != 1080 or not all(k.endswith("_N") for k in features):
        raise ValueError("Unexpected UCI mouse-protein table schema")
    values = np.array([[float(r[k]) if r[k] else np.nan for k in features] for r in rows])
    if np.any(np.isinf(values)):
        raise ValueError("Infinite protein value")
    # Target selection does not use the target genotype labels.
    target = np.array([r["Behavior"] == "S/C" and r["Treatment"] == "Saline" for r in rows])
    background = np.array([r["Behavior"] == "C/S" and r["Treatment"] == "Saline"
                           and r["Genotype"] == "Control" for r in rows])
    identifiers = np.array([r["MouseID"] for r in rows])
    genotypes = np.array([r["Genotype"] for r in rows])
    return dict(raw_target=values[target], raw_background=values[background],
                feature_names=np.array(features), target_ids=identifiers[target],
                background_ids=identifiers[background], target_labels=genotypes[target])


def standardize(data):
    """Author settings: zero imputation, centering, population std per feature.

    A constant column is set to zero. Target and background each use their own
    means and scales, as in CPCA(standardize=True), not a jointly fitted scaler.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2 or np.any(np.isinf(data)):
        raise ValueError("Expected >=2 observations and features, finite or missing values")
    filled = np.where(np.isnan(data), 0., data)
    mean = filled.mean(axis=0)
    centered = filled-mean
    scale = centered.std(axis=0, ddof=0)
    scaled = np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 0)
    return scaled, mean, scale


def covariance_factor(observations):
    """Thin SVD without truncation: F=P*S preserves XX* in exact arithmetic.

    X=observations.T/sqrt(n-1)=P*S*Q*. Dropping only the right orthonormal
    factor Q reduces the number of columns, not the covariance rank.
    """
    x = observations.T / np.sqrt(observations.shape[0]-1)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return x, u*s


def orient_columns(basis):
    """Fix display signs by loadings, without inspecting observation labels."""
    basis = basis.copy()
    for i in range(basis.shape[1]):
        largest = np.argmax(np.abs(basis[:, i]))
        if basis[largest, i] < 0:
            basis[:, i] *= -1
    return basis


def fit_cpca(raw_target, raw_background, alpha=ALPHA):
    """Fit PCA and 2D cPCA; no labels are accepted by this function."""
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    target, mean_x, scale_x = standardize(raw_target)
    background, mean_y, scale_y = standardize(raw_background)
    if target.shape[1] != background.shape[1]:
        raise ValueError("Target and background must use the same features")
    x, fx = covariance_factor(target)
    y, fy = covariance_factor(background)
    a = np.column_stack((fx, np.sqrt(alpha)*fy))
    p = fx.shape[1]
    signs = np.r_[np.ones(p), -np.ones(fy.shape[1])]
    result = hsvd(a, p=p)  # Always the general algorithm, including rank deficiency.
    signed_values = np.diag((result.Sigma*signs) @ result.Sigma.T)
    order = np.argsort(-signed_values, kind="stable")
    basis = orient_columns(result.U[:, order[:2]])
    h = x @ x.T-alpha*(y @ y.T)
    reference_values, reference_vectors = np.linalg.eigh(h)
    reference_basis = reference_vectors[:, -2:]
    pca_u, pca_s, _ = np.linalg.svd(x, full_matrices=False)
    pca_basis = orient_columns(pca_u[:, :2])
    projector_error = np.linalg.norm(basis @ basis.T-reference_basis @ reference_basis.T, "fro")
    norm_h = np.linalg.norm(h, "fro")
    gap = float(reference_values[-2]-reference_values[-3]) if len(reference_values) > 2 else None
    projector_is_unique = bool(gap is None or gap > 1e-10*max(norm_h, 1.))
    summary = dict(
        alpha=alpha, features=target.shape[1], target_records=len(target),
        background_records=len(background), original_factor_shape=[x.shape[0], x.shape[1]+y.shape[1]],
        compressed_factor_shape=list(a.shape), structure=asdict(result.structure),
        missing_target=int(np.isnan(raw_target).sum()), missing_background=int(np.isnan(raw_background).sum()),
        selected_signed_eigenvalues=signed_values[order[:2]].tolist(),
        reference_largest_eigenvalues=reference_values[-2:][::-1].tolist(),
        pca_explained_variance_ratios=(pca_s[:2]**2 / np.sum(pca_s**2)).tolist(),
        cpca_contrast_trace=float(np.trace(basis.T @ h @ basis)),
        pca_contrast_trace=float(np.trace(pca_basis.T @ h @ pca_basis)),
        projector_error=float(projector_error),
        projector_is_unique=projector_is_unique, second_to_third_eigenvalue_gap=gap,
        optimal_trace_error_absolute=float(abs(np.trace(basis.T @ h @ basis)-reference_values[-2:].sum())),
        eigen_residual_relative=float(np.linalg.norm(h @ basis-basis*signed_values[order[:2]], "fro")/max(norm_h, 1.)),
        covariance_compression_error_relative=float(np.linalg.norm((a*signs) @ a.T-h, "fro")/max(norm_h, 1.)),
        spectrum_error_absolute=float(np.max(np.abs(np.sort(signed_values)-reference_values))),
        hsvd_diagnostics=asdict(result.diagnostics),
    )
    if max(summary["eigen_residual_relative"], summary["covariance_compression_error_relative"],
           summary["optimal_trace_error_absolute"]/max(norm_h, 1.)) > 1e-9:
        raise AssertionError("HSVD/covariance/eigh verification failed")
    if projector_is_unique and summary["projector_error"] > 1e-9:
        raise AssertionError("Unique dominant subspaces disagree")
    arrays = dict(target_standardized=target, background_standardized=background,
                  target_mean=mean_x, background_mean=mean_y,
                  target_scale=scale_x, background_scale=scale_y,
                  X=x, Y=y, factor_X=fx, factor_Y=fy, A=a, signs=signs, H=h,
                  U=result.U, Sigma=result.Sigma, V=result.V,
                  signed_values=signed_values, cpca_basis=basis, pca_basis=pca_basis,
                  cpca_scores=target @ basis, pca_scores=target @ pca_basis,
                  reference_eigenvalues=reference_values)
    return summary, arrays


def mouse_counts(inputs):
    # The suffix _1,...,_15 identifies a technical measurement, not a new mouse.
    counts = {}
    for group in ("target", "background"):
        mice, per_mouse = np.unique([s.rsplit("_", 1)[0] for s in inputs[f"{group}_ids"]], return_counts=True)
        counts[f"{group}_mice"] = len(mice)
        counts[f"{group}_records_per_mouse"] = sorted(set(per_mouse.tolist()))
    counts["target_genotype_records"] = {str(k): int(np.sum(inputs["target_labels"] == k))
                                          for k in np.unique(inputs["target_labels"])}
    return counts


def create_figures(inputs, arrays, output):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hsvd-matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "hsvd-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ("#2166ac", "#c34e18")
    groups = ("Control", "Ts65Dn")
    labels = inputs["target_labels"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), layout="constrained")
    display_labels = ("Контрольный генотип (135 измерений, 9 мышей)", "Ts65Dn (135 измерений, 9 мышей)")
    for axis, method, title in zip(axes, ("pca", "cpca"), ("PCA", r"cPCA через HSVD ($\alpha=2$)")):
        scores = arrays[f"{method}_scores"]
        for group, color, display_label in zip(groups, colors, display_labels):
            mask = labels == group
            axis.scatter(*scores[mask].T, s=15, color=color, alpha=.75, linewidths=0,
                         label=display_label)
        axis.set(title=title, xlabel="PC1" if method == "pca" else "cPC1",
                 ylabel="PC2" if method == "pca" else "cPC2")
        axis.grid(alpha=.18)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="best", fontsize=8)
    for extension in ("png", "pdf"):
        kw = {"dpi": 190} if extension == "png" else {"metadata": {"CreationDate": None, "ModDate": None}}
        fig.savefig(output / f"pca_vs_cpca.{extension}", **kw)
    plt.close(fig)


def source_hashes():
    return {name: sha256(ROOT/name) for name in SOURCES}


def run(output=DEFAULT_OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # --output is also usable offline: take the verified repository cache.
    if not (output/"data.csv").exists():
        (output/"data.csv").write_bytes((DEFAULT_OUTPUT/"data.csv").read_bytes())
    if not (output/"PROVENANCE.md").exists():
        (output/"PROVENANCE.md").write_bytes((DEFAULT_OUTPUT/"PROVENANCE.md").read_bytes())
    before = source_hashes()
    inputs = read_dataset(output/"data.csv")
    summary, arrays = fit_cpca(inputs["raw_target"], inputs["raw_background"])
    summary.update(mouse_counts(inputs))
    np.savez_compressed(output/"inputs.npz", **inputs)
    np.savez_compressed(output/"computed.npz", **arrays)
    with (output/"projection.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["record_id", "mouse_id", "genotype", "PC1", "PC2", "cPC1", "cPC2"])
        for rid, label, pca, cpca in zip(inputs["target_ids"], inputs["target_labels"], arrays["pca_scores"], arrays["cpca_scores"]):
            writer.writerow([rid, rid.rsplit("_", 1)[0], label, *pca, *cpca])
    create_figures(inputs, arrays, output)
    if source_hashes() != before:
        raise RuntimeError("Source changed during the run")
    summary.update(protocol="cpca-real-v1", dataset_url=DATA_URL, dataset_sha256=DATA_SHA256,
                   python=platform.python_version(), numpy=np.__version__, source_hashes=before)
    summary["artifact_hashes"] = {name: sha256(output/name) for name in ARTIFACTS}
    (output/"results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    return summary


def check(output=DEFAULT_OUTPUT):
    """Offline replay, cache/source integrity and independent factor identities."""
    output = Path(output)
    saved = json.loads((output/"results.json").read_text(encoding="utf-8"))
    for key, expected in {"protocol": "cpca-real-v1", "dataset_url": DATA_URL,
                          "dataset_sha256": DATA_SHA256, "alpha": ALPHA}.items():
        if saved[key] != expected:
            raise AssertionError(f"Provenance metadata mismatch: {key}")
    if saved["source_hashes"] != source_hashes():
        raise AssertionError("Source hash mismatch; regenerate before checking")
    if set(saved["artifact_hashes"]) != set(ARTIFACTS):
        raise AssertionError("Required artifact inventory differs")
    for name, expected in saved["artifact_hashes"].items():
        if sha256(output/name) != expected:
            raise AssertionError(f"Artifact hash mismatch: {name}")
    inputs = read_dataset(output/"data.csv")
    with np.load(output/"inputs.npz", allow_pickle=False) as cached:
        if set(cached.files) != set(inputs):
            raise AssertionError("Cached input fields differ")
        for key, value in inputs.items():
            np.testing.assert_array_equal(cached[key], value)
    summary, arrays = fit_cpca(inputs["raw_target"], inputs["raw_background"], alpha=saved["alpha"])
    for key, value in mouse_counts(inputs).items():
        if saved[key] != value:
            raise AssertionError(f"Mouse/record metadata mismatch: {key}")
    with np.load(output/"computed.npz", allow_pickle=False) as old:
        for key in ("H", "cpca_scores", "pca_scores", "reference_eigenvalues"):
            np.testing.assert_allclose(old[key], arrays[key], rtol=5e-9, atol=5e-10)
        # Full V has arbitrary complementary bases; verify its defining identities
        # instead of requiring those basis entries to replay bit for bit.
        u, sigma, v, a, signs = (old[k] for k in ("U", "Sigma", "V", "A", "signs"))
        np.testing.assert_allclose(u @ sigma @ v.T, a, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-9)
        np.testing.assert_allclose(v.T @ (signs[:, None]*v), np.diag(signs), atol=1e-8)
        np.testing.assert_allclose((a*signs) @ a.T, old["H"], atol=1e-9)
    for key, value in summary.items():
        if key == "hsvd_diagnostics":
            for metric, number in value.items():
                np.testing.assert_allclose(saved[key][metric], number, rtol=5e-9, atol=5e-10)
        elif isinstance(value, (dict, bool)) or value is None:
            if saved[key] != value:
                raise AssertionError(f"Result metadata mismatch: {key}")
        else:
            np.testing.assert_allclose(saved[key], value, rtol=5e-9, atol=5e-10)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify and replay cached artifacts without network or writes")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = check(args.output) if args.check else run(args.output)
    print(json.dumps({"status": "PASS", "projector_error": summary["projector_error"],
                      "selected_signed_eigenvalues": summary["selected_signed_eigenvalues"]}, indent=2))


if __name__ == "__main__":
    main()
