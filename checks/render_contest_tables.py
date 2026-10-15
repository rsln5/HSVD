"""Render experimental tables from a source- and input-hash-verified protocol.

Run from the project root:
    python3 -m checks.render_contest_tables
    python3 -m checks.render_contest_tables --check
    python3 -m checks.render_contest_tables --export output/experiment_tables.tex

The default output is a standalone TeX fragment. Importing this module does
not read or modify files, and the renderer never edits the manuscript.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from checks.contest_experiments import PROTOCOL, SOURCE_PATHS, jacobi_applicability

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("reconstruction_error", "unitarity_error", "j_unitarity_error")


def load_protocol(path: Path, root: Path = ROOT) -> dict:
    """Reject stale sources or a changed saved input archive before rendering."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("protocol") != PROTOCOL:
        raise ValueError(f"Expected {PROTOCOL}; run the experiments again")
    required = set(SOURCE_PATHS)
    if not required.issubset(data.get("sha256", {})):
        raise ValueError("Incomplete source manifest; run the experiments again")
    for source, digest in data["sha256"].items():
        if hashlib.sha256((root / source).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Stale protocol: {source}; run the experiments again")
    if hashlib.sha256((path.parent / "inputs.npz").read_bytes()).hexdigest() != data.get("inputs_sha256"):
        raise ValueError("Saved inputs do not match the protocol; run the experiments again")
    for group in ("structural", "comparison", "conditioning", "thresholds"):
        if not data.get(group):
            raise ValueError(f"Missing experimental group: {group}")
    if any(row["status"] != "ok" for row in data["structural"]):
        raise ValueError("A structural experiment failed")
    # Recheck the baseline domain from the saved matrices, rather than
    # trusting a claimed rank or silently dropping unsupported examples.
    with np.load(path.parent / "inputs.npz", allow_pickle=False) as inputs:
        for group in ("structural", "comparison", "conditioning", "thresholds"):
            for row in data[group]:
                if row.get("input_key") not in inputs:
                    raise ValueError("Missing saved experimental input")
                a = inputs[row["input_key"]]
                if list(a.shape) != row.get("shape"):
                    raise ValueError("Saved input shape does not match the protocol")
                applicable = jacobi_applicability(a)["applicable"]
                general = row["general"] if group in ("comparison", "conditioning") else row
                baseline = row["jacobi"]
                for method, record in (("general", general), ("jacobi", baseline)):
                    status = record.get("status")
                    if method == "jacobi" and not applicable:
                        if status != "not_applicable" or record.get("elapsed_seconds") is not None:
                            raise ValueError("Jacobi must not run outside its full-column-rank domain")
                        continue
                    if status not in ("ok", "failed"):
                        raise ValueError("An applicable method was not run")
                    elapsed = record.get("elapsed_seconds")
                    if elapsed is None or not np.isfinite(elapsed) or elapsed < 0:
                        raise ValueError("Missing or invalid computation duration")
                    if status == "failed" and not record.get("reason"):
                        raise ValueError("A failed attempt must retain its reason")
                    if status == "ok" and any(
                            key not in record.get("metrics", {})
                            or not np.isfinite(record["metrics"][key])
                            or record["metrics"][key] < 0 for key in METRICS):
                        raise ValueError("Missing or nonfinite successful diagnostics")
                if group == "comparison":
                    if not applicable:
                        raise ValueError("A timing input is outside the common method domain")
                    for record in (general, baseline):
                        attempts = record.get("attempts", [])
                        for index, attempt in enumerate(attempts):
                            duration = attempt.get("elapsed_seconds")
                            if (attempt.get("repetition") != index
                                    or attempt.get("status") not in ("ok", "failed")
                                    or duration is None or not np.isfinite(duration) or duration < 0
                                    or (attempt["status"] == "failed" and not attempt.get("reason"))):
                                raise ValueError("Incomplete timing attempt or failure duration")
                        if record.get("time_seconds") != [x["elapsed_seconds"] for x in attempts
                                                          if x["status"] == "ok"]:
                            raise ValueError("Successful timing samples do not match the attempt log")
                        if record["status"] == "ok":
                            durations = [x["elapsed_seconds"] for x in attempts]
                            if (len(attempts) != data["settings"]["timing_repetitions"]
                                    or any(x["status"] != "ok" for x in attempts)
                                    or any(not np.isfinite(t) or t < 0 for t in durations)
                                    or durations != record.get("time_seconds")
                                    or not np.isclose(record.get("median_ms", np.nan),
                                                      statistics.median(durations)*1000)):
                                raise ValueError("Incomplete or inconsistent timing attempts")
                        elif record.get("median_ms") is not None:
                            raise ValueError("A failed comparison cannot report a successful median")
    return data


def sci(value):
    if value == 0:
        return "$0$"
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"${mantissa}\\cdot10^{{{int(exponent)}}}$"


def table(caption, label, fmt, head, rows):
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering\small", r"\caption{" + caption + "}",
        r"\label{" + label + "}", r"\begin{tabular}{@{}" + fmt + "@{}}",
        r"\toprule", head + r"\\", r"\midrule", *[row + r"\\" for row in rows],
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]) + "\n"


def render_tables(data):
    groups = [
        ("Явные примеры", [x for x in data["structural"] if x["group"] == "analytic"]),
        ("Сгенерированные вещественные", [x for x in data["structural"]
                                          if x["group"] == "generated" and not x["complex_data"]]),
        ("Сгенерированные комплексные", [x for x in data["structural"]
                                         if x["group"] == "generated" and x["complex_data"]]),
    ]
    rows = [name + " & " + str(len(cases)) + " & " + " & ".join(
        sci(max(x["metrics"][key] for x in cases)) for key in METRICS)
        for name, cases in groups if cases]
    structural = table("Проверки заданной при генерации структуры и невязок",
                       "tab:structural", "lrrrr",
                       r"Группа & Число & $\max\eta_A$ & $\max\eta_U$ & $\max\eta_V$", rows)

    rows = []
    for m, n in sorted({tuple(x["shape"]) for x in data["comparison"]}):
        cases = [x for x in data["comparison"] if x["shape"] == [m, n]]
        for method, name in (("general", "Общий"), ("jacobi", "Прототип Якоби")):
            if any(x[method]["status"] != "ok" for x in cases):
                rows.append(f"${m}\\times {n}$ & {name} & отказ & --- & --- & ---")
            else:
                median = statistics.median(x[method]["median_ms"] for x in cases)
                rows.append(f"${m}\\times {n}$ & {name} & {median:.3f} & " + " & ".join(
                    sci(max(x[method]["metrics"][key] for x in cases)) for key in METRICS))
    comparison = table("Полные факторы на матрицах полного столбцового ранга: медиана медианных времён в миллисекундах и максимальные невязки",
                       "tab:comparison", "llrrrr",
                       r"Размер & Метод & Время & $\max\eta_A$ & $\max\eta_U$ & $\max\eta_V$", rows)

    rows = []
    for case in data["conditioning"]:
        fields = [sci(case[method]["max_relative_value_error_stored_input"])
                  if case[method]["status"] == "ok" else
                  ("не применим" if case[method]["status"] == "not_applicable" else "отказ")
                  for method in ("general", "jacobi")]
        fields.extend(sci(case[method]["metrics"]["reconstruction_error"])
                      if case[method]["status"] == "ok" else "---"
                      for method in ("general", "jacobi"))
        rows.append(sci(case["sigma_ratio"]) + " & " + " & ".join(fields))
    conditioning = table("Разброс заданных гиперболических чисел: ошибки относительно эталона для сохранённого входа, вычисленного с 80 десятичными знаками",
                         "tab:spectrum", "rrrrr",
                         r"$\kappa_\Sigma$ & $e_\sigma$, общий & $e_\sigma$, Якоби & $\eta_A$, общий & $\eta_A$, Якоби", rows)

    rows = []
    thresholds = sorted({x["spectral_rtol"] for x in data["thresholds"]}, reverse=True)
    for delta in sorted({x["delta"] for x in data["thresholds"]}):
        fields = []
        for threshold in thresholds:
            selected = [x for x in data["thresholds"]
                        if x["delta"] == delta and x["spectral_rtol"] == threshold]
            if len(selected) != 1:
                raise ValueError("Threshold grid contains missing or duplicate entries")
            case = selected[0]
            fields.append("$j_{\\rm num}=" + str(case["structure"]["j"]) + "$"
                          if case["status"] == "ok" else "отказ")
        rows.append(sci(delta) + " & " + " & ".join(fields))
    threshold_head = r"$\delta$ & " + " & ".join(
        r"$\tau_C=" + sci(value)[1:-1] + "$" for value in thresholds)
    threshold_table = table(r"Зависимость результата для $A_\delta=(1\;1+\delta)$ от порога",
                            "tab:threshold", "r" * (1 + len(thresholds)), threshold_head, rows)
    return "\n".join((structural, comparison, conditioning, threshold_table))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--export", type=Path, default=ROOT / "output/experiment_tables.tex",
                        help="Standalone TeX fragment to write or verify")
    parser.add_argument("--protocol", type=Path, default=ROOT / "results/contest/results.json")
    args = parser.parse_args()
    try:
        rendered = render_tables(load_protocol(args.protocol))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    if args.check:
        if not args.export.exists() or args.export.read_text(encoding="utf-8") != rendered:
            parser.exit(1, "Exported tables do not match the saved protocol\n")
        print("Exported tables match the source- and input-hash-verified protocol.")
    else:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(rendered, encoding="utf-8")
        print(f"Exported four tables to {args.export}.")


if __name__ == "__main__":
    main()
