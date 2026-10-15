"""The presentation range must not hide failures within that range."""
from copy import deepcopy
import json
import shutil
import sys

import pytest

from checks import render_paper_tables as paper
from checks.contest_experiments import SOURCE_PATHS
from checks.render_contest_tables import ROOT, load_protocol, render_tables


@pytest.fixture
def data():
    return json.loads((ROOT / "results/contest/results.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("status", ["ok", "failed"])
def test_range_selection_is_independent_of_status_and_preserves_input(data, status):
    outside = next(case for case in data["conditioning"] if case["sigma_ratio"] == 1e8)
    outside["general"] = deepcopy(outside["jacobi"])
    outside["general"]["status"] = status
    before = deepcopy(data)
    result = paper.render_paper_tables(data)
    assert r"$1.00\cdot10^{8}$" not in result
    assert r"$1.00\cdot10^{6}$" in result
    assert r"\kappa_\Sigma\le10^6" in result
    assert data == before
    # Structural, timing and threshold tables retain their complete output.
    original_tables = render_tables(data).split(r"\end{table}")
    paper_tables = result.split(r"\end{table}")
    for index in (0, 1, 3):
        assert paper_tables[index] == original_tables[index]


def test_failure_within_the_reported_range_remains_visible(data):
    case = next(case for case in data["conditioning"] if case["sigma_ratio"] == 1e6)
    case["general"]["status"] = "failed"
    assert r"$1.00\cdot10^{6}$ & отказ &" in paper.render_paper_tables(data)


@pytest.mark.parametrize("damage", ["missing_inputs", "stale_inputs", "stale_source"])
def test_cli_validates_raw_protocol_before_export(tmp_path, monkeypatch, damage):
    for relative in (*SOURCE_PATHS, "results/contest/results.json", "results/contest/inputs.npz"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    if damage == "missing_inputs":
        (tmp_path / "results/contest/inputs.npz").unlink()
    else:
        target = "hsvd/core.py" if damage == "stale_source" else "results/contest/inputs.npz"
        (tmp_path / target).write_bytes(b"changed")
    exported = tmp_path / "tables.tex"
    monkeypatch.setattr(sys, "argv", ["render_paper_tables", "--protocol",
                        str(tmp_path / "results/contest/results.json"), "--export", str(exported)])
    monkeypatch.setattr(paper, "load_protocol", lambda path: load_protocol(path, tmp_path))
    with pytest.raises(SystemExit) as exc:
        paper.main()
    assert exc.value.code == 1
    assert not exported.exists()
