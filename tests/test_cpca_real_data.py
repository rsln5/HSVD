"""Independent mathematical and provenance checks of the real-data example."""
import json
from copy import deepcopy

import numpy as np
import pytest

from examples.cpca_real_data import (
    ALPHA, DEFAULT_OUTPUT, covariance_factor, fit_cpca, mouse_counts,
    read_dataset, standardize,
)


@pytest.fixture(scope="module")
def inputs():
    return read_dataset(DEFAULT_OUTPUT / "data.csv")


@pytest.fixture(scope="module")
def fitted(inputs):
    return fit_cpca(inputs["raw_target"], inputs["raw_background"])


def test_real_data_counts_and_repeated_measurements(inputs):
    assert inputs["raw_target"].shape == (270, 77)
    assert inputs["raw_background"].shape == (135, 77)
    assert mouse_counts(inputs) == dict(target_mice=18, background_mice=9,
                                      target_records_per_mouse=[15], background_records_per_mouse=[15],
                                      target_genotype_records={"Control": 135, "Ts65Dn": 135})
    assert not set(inputs["target_ids"]) & set(inputs["background_ids"])
    assert np.isnan(inputs["raw_target"]).any()


def test_zero_imputation_and_population_standardization():
    raw = np.array([[1., np.nan, 3.], [3., 2., 3.], [5., 4., 3.]])
    z, mean, scale = standardize(raw)
    np.testing.assert_allclose(mean, [3., 2., 3.])
    np.testing.assert_allclose(scale, [np.sqrt(8/3), np.sqrt(8/3), 0.])
    np.testing.assert_allclose(z[:, :2].var(axis=0), 1.)
    np.testing.assert_array_equal(z[:, 2], 0.)


def test_svd_compression_preserves_full_covariance_including_deficiency():
    observations = np.array([[1., 2., 3.], [2., 4., 6.], [-1., -2., -3.], [0., 0., 0.]])
    x, factor = covariance_factor(observations)
    assert factor.shape == (3, 3)  # No rank truncation; all singular values retained.
    np.testing.assert_allclose(factor @ factor.T, observations.T @ observations/3, atol=1e-13)
    np.testing.assert_allclose(factor @ factor.T, x @ x.T, atol=1e-13)


def test_hsvd_matches_independently_formed_covariances(inputs, fitted):
    summary, data = fitted
    # Independent preprocessing, without calling the implementation's scaler.
    z = []
    for key in ("raw_target", "raw_background"):
        filled = np.nan_to_num(inputs[key], nan=0.)
        z.append((filled-filled.mean(0))/filled.std(0, ddof=0))
    covariance = np.cov(z[0], rowvar=False)-ALPHA*np.cov(z[1], rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    basis = data["cpca_basis"]
    np.testing.assert_allclose(data["H"], covariance, atol=2e-13)
    np.testing.assert_allclose(basis @ basis.T, vectors[:, -2:] @ vectors[:, -2:].T, atol=1e-11)
    np.testing.assert_allclose(np.trace(basis.T @ covariance @ basis), values[-2:].sum(), atol=1e-11)
    np.testing.assert_allclose(summary["selected_signed_eigenvalues"], values[-2:][::-1], atol=1e-11)
    assert summary["compressed_factor_shape"] == [77, 154]
    assert summary["structure"]["rank"] < 77  # Real duplicated protein feature retained.
    assert summary["projector_error"] < 1e-10


def test_target_labels_cannot_change_fitted_directions(inputs, fitted):
    original = inputs["target_labels"].copy()
    inputs["target_labels"] = original[::-1]
    try:
        _, other = fit_cpca(inputs["raw_target"], inputs["raw_background"])
    finally:
        inputs["target_labels"] = original
    np.testing.assert_allclose(other["cpca_basis"], fitted[1]["cpca_basis"], atol=1e-12)


def test_zero_contrast_recovers_pca_subspace(inputs):
    _, data = fit_cpca(inputs["raw_target"], inputs["raw_background"], alpha=0.)
    cpca, pca = data["cpca_basis"], data["pca_basis"]
    np.testing.assert_allclose(cpca @ cpca.T, pca @ pca.T, atol=1e-11)


def test_two_largest_signed_values_are_selected_even_when_negative():
    # Identical correlated covariances yield distinct negative values. Largest
    # modulus would choose the wrong two-dimensional space.
    covariance = np.array([[1., .2, .1], [.2, 1., .4], [.1, .4, 1.]])
    factor = np.linalg.cholesky(covariance).T
    raw = np.r_[factor, -factor]
    summary, data = fit_cpca(raw, raw, alpha=2.)
    assert np.all(np.array(summary["selected_signed_eigenvalues"]) < 0)
    values, vectors = np.linalg.eigh(-1.2*covariance)
    np.testing.assert_allclose(summary["selected_signed_eigenvalues"], values[-2:][::-1], atol=1e-12)
    basis = data["cpca_basis"]
    np.testing.assert_allclose(basis @ basis.T, vectors[:, -2:] @ vectors[:, -2:].T, atol=1e-12)
    np.testing.assert_allclose(data["cpca_basis"].T @ data["cpca_basis"], np.eye(2), atol=1e-12)


def test_repeated_boundary_eigenvalue_does_not_require_unique_projector():
    raw = np.r_[np.eye(3), -np.eye(3)]
    summary, _ = fit_cpca(raw, raw, alpha=2.)
    assert not summary["projector_is_unique"]
    assert summary["optimal_trace_error_absolute"] < 1e-12


def test_corrupted_dataset_is_rejected(tmp_path):
    path = tmp_path / "data.csv"
    path.write_bytes((DEFAULT_OUTPUT / "data.csv").read_bytes()+b"\n")
    with pytest.raises(ValueError, match="checksum"):
        read_dataset(path)


def test_offline_replay_checks_inputs_and_output_integrity(tmp_path, monkeypatch):
    from examples import cpca_real_data as example

    def placeholder_figures(inputs, arrays, output):
        # Numerical/protocol tests do not require the optional plotting dependency.
        for extension in ("png", "pdf"):
            (output/f"pca_vs_cpca.{extension}").write_bytes(b"test-only figure")

    monkeypatch.setattr(example, "create_figures", placeholder_figures)
    example.run(tmp_path)
    assert example.check(tmp_path)["projector_error"] < 1e-10
    path = tmp_path/"results.json"
    metadata = json.loads(path.read_text())
    metadata["target_mice"] = 270  # Confusing measurements with independent animals.
    path.write_text(json.dumps(metadata))
    with pytest.raises(AssertionError, match="Mouse/record metadata"):
        example.check(tmp_path)
    metadata["target_mice"] = 18
    path.write_text(json.dumps(metadata))
    for key, invalid in (("artifact_hashes", {}), ("dataset_sha256", "wrong"),
                         ("dataset_url", "wrong"), ("protocol", "wrong"),
                         ("features", 3), ("structure", {"rank": 100})):
        changed = deepcopy(metadata)
        changed[key] = invalid
        path.write_text(json.dumps(changed))
        with pytest.raises(AssertionError):
            example.check(tmp_path)
    path.write_text(json.dumps(metadata))
    (tmp_path/"projection.csv").write_text("corrupted")
    with pytest.raises(AssertionError, match="Artifact hash mismatch"):
        example.check(tmp_path)


@pytest.mark.parametrize("alpha", [-1., np.nan, np.inf])
def test_invalid_alpha_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        fit_cpca(np.eye(3), np.eye(3), alpha=alpha)
