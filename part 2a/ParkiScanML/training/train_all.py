"""Train and evaluate research models on the attached public datasets with participant-grouped CV.

    python -m training.train_all                 # all datasets found under PARKISCAN_DATA_DIR
    python -m training.train_all --only spiral   # one dataset

Rules enforced here:
* Splits are grouped by participant (StratifiedGroupKFold / GroupKFold / leave-one-subject-out);
  no participant ever appears in both train and test folds.
* Every model is compared with a trivial baseline (DummyClassifier / DummyRegressor).
* Metrics are computed on pooled out-of-fold predictions, plus participant-level aggregation where a
  participant has several recordings, with a participant-bootstrap 95% CI for ROC AUC.
* Outputs: models/<dataset>.joblib (+ .json metadata), reports/training_report.{json,md}.
These are research benchmarks on small public datasets. They are NOT clinically validated and must
not be used to diagnose anyone.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, \
    RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.config import get_settings
from parkiscan.datasets.loaders import LOADERS, available

warnings.filterwarnings("ignore")
DISCLAIMER = ("Research benchmark on a small public dataset with participant-grouped cross-validation. "
              "Not clinically validated; not for diagnosis or screening decisions.")


def classifiers(seed):
    imp = SimpleImputer(strategy="median")
    return {
        "dummy_most_frequent": make_pipeline(SimpleImputer(strategy="median"), DummyClassifier(strategy="prior")),
        "logistic_regression": make_pipeline(clone(imp), StandardScaler(),
                                             LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")),
        "random_forest": make_pipeline(clone(imp), RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                                          class_weight="balanced", random_state=seed,
                                                                          n_jobs=-1)),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=200, min_samples_leaf=3,
                                                                 class_weight="balanced", random_state=seed),
    }


def regressors(seed):
    imp = SimpleImputer(strategy="median")
    return {
        "dummy_mean": make_pipeline(clone(imp), DummyRegressor()),
        "ridge": make_pipeline(clone(imp), StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": make_pipeline(clone(imp), RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                                                         random_state=seed, n_jobs=-1)),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.08, max_iter=200, min_samples_leaf=5,
                                                                random_state=seed),
    }


def _bootstrap_auc(y, p, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        pick = rng.choice(ug, size=ug.size, replace=True)
        ii = np.concatenate([idx[g] for g in pick])
        if len(np.unique(y[ii])) == 2:
            vals.append(roc_auc_score(y[ii], p[ii]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None]


def _cls_metrics(y, p, groups):
    yhat = (p >= 0.5).astype(int)
    m = {"n": int(y.size), "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
         "f1": float(f1_score(y, yhat, zero_division=0)),
         "sensitivity": float(np.mean(yhat[y == 1] == 1)) if np.any(y == 1) else None,
         "specificity": float(np.mean(yhat[y == 0] == 0)) if np.any(y == 0) else None}
    if len(np.unique(y)) == 2:
        m["roc_auc"] = float(roc_auc_score(y, p))
        m["roc_auc_95ci_participant_bootstrap"] = _bootstrap_auc(y, p, groups)
    return m


def eval_classification(X, y, groups, n_splits, repeats, loso=False):
    y = y.to_numpy()
    g = groups.to_numpy()
    results = {}
    for name in classifiers(0):
        rep_auc, last_oof = [], None
        for r in range(1 if loso else repeats):
            oof = np.full(y.size, np.nan)
            splitter = LeaveOneGroupOut() if loso else StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=r)
            for tr, te in splitter.split(X, y, g):
                if len(np.unique(y[tr])) < 2:
                    continue
                mdl = clone(classifiers(r)[name]).fit(X.iloc[tr], y[tr])
                oof[te] = mdl.predict_proba(X.iloc[te])[:, 1]
            ok = np.isfinite(oof)
            if len(np.unique(y[ok])) == 2:
                rep_auc.append(roc_auc_score(y[ok], oof[ok]))
            last_oof = oof
        ok = np.isfinite(last_oof)
        rec = {"recording_level": _cls_metrics(y[ok], last_oof[ok], g[ok]),
               "roc_auc_across_repeats": {"mean": float(np.mean(rep_auc)), "sd": float(np.std(rep_auc)),
                                          "repeats": len(rep_auc)} if rep_auc else None}
        if pd.Series(g).duplicated().any() and not loso:
            df = pd.DataFrame({"g": g[ok], "y": y[ok], "p": last_oof[ok]}).groupby("g").agg(y=("y", "first"), p=("p", "mean"))
            rec["participant_level"] = _cls_metrics(df["y"].to_numpy(), df["p"].to_numpy(), df.index.to_numpy())
        if loso:
            per = {}
            for gg in np.unique(g):
                mm = (g == gg) & ok
                if len(np.unique(y[mm])) == 2:
                    per[str(gg)] = float(roc_auc_score(y[mm], last_oof[mm]))
            rec["per_subject_auc"] = per
        results[name] = rec
    return results


def eval_regression(X, y, groups, n_splits):
    y = y.to_numpy()
    g = groups.to_numpy()
    out = {}
    for name in regressors(0):
        oof = np.full(y.size, np.nan)
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, g):
            oof[te] = clone(regressors(0)[name]).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        out[name] = {"recording_level": {"n": int(y.size), "mae": float(mean_absolute_error(y, oof)),
                                         "r2": float(r2_score(y, oof))}}
    return out


def quality_flags(entry: dict) -> list[str]:
    """Automatic sanity flags so implausible or useless results are never presented as findings."""
    res, best, flags = entry["results"], entry["best_model"], []
    if entry["task"] == "regression":
        dummy = next(v for k, v in res.items() if k.startswith("dummy"))["recording_level"]["mae"]
        if res[best]["recording_level"]["mae"] >= dummy:
            flags.append("NO_BETTER_THAN_BASELINE: best model does not beat predicting the training mean")
        return flags
    lvl = res[best].get("participant_level") or res[best]["recording_level"]
    auc, ci = lvl.get("roc_auc") or 0, (lvl.get("roc_auc_95ci_participant_bootstrap") or [None, None])
    if auc >= 0.98 and entry["n_participants"] < 100:
        flags.append("SUSPICIOUSLY_HIGH: near-perfect separation on a tiny sample usually means a confound or "
                     "label leakage (recording setup, rendering, task duration); do not rely on this model")
    if ci[0] is not None and ci[0] <= 0.5:
        flags.append("CI_INCLUDES_CHANCE: the 95% participant-bootstrap CI of ROC AUC includes 0.5")
    if entry["n_participants"] < 50:
        flags.append(f"SMALL_SAMPLE: only {entry['n_participants']} participants")
    return flags


def pick_best(res, task):
    cands = {k: v for k, v in res.items() if not k.startswith("dummy")}
    if task == "regression":
        return min(cands, key=lambda k: cands[k]["recording_level"]["mae"])
    key = lambda k: (cands[k].get("participant_level") or cands[k]["recording_level"]).get("roc_auc", 0)
    return max(cands, key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    s = get_settings()
    s.models_dir.mkdir(parents=True, exist_ok=True)
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    avail = available(s.data_dir)
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "data_dir": str(s.data_dir),
              "disclaimer": DISCLAIMER, "datasets": {}}
    for name, loader in LOADERS.items():
        if a.only and name not in a.only:
            continue
        if not avail.get(name):
            report["datasets"][name] = {"status": "not found on disk — skipped"}
            print(f"[train] {name}: not found, skipped")
            continue
        t0 = time.time()
        X, y, g, meta = loader(s.data_dir)
        n_groups = g.nunique()
        if meta["task"] == "regression":
            res = eval_regression(X, y, g, n_splits=min(5, n_groups))
        else:
            loso = name == "daphnet"
            n_min = int(min(np.bincount(pd.Series(y).groupby(g).first().astype(int))))
            res = eval_classification(X, y, g, n_splits=max(2, min(5, n_min)), repeats=a.repeats, loso=loso)
        best = pick_best(res, meta["task"])
        pos = None if meta["task"] == "regression" else int(pd.Series(y).groupby(g).max().sum())
        entry = {"status": "evaluated", **meta, "n_rows": int(len(y)), "n_participants": int(n_groups),
                 "n_positive_participants": pos, "features": list(X.columns),
                 "cv": "leave-one-subject-out" if name == "daphnet" else
                 ("GroupKFold(5)" if meta["task"] == "regression" else f"StratifiedGroupKFold x{a.repeats} repeats"),
                 "results": res, "best_model": best, "seconds": round(time.time() - t0, 1)}
        entry["quality_flags"] = quality_flags(entry)
        entry["usable_for_research_scores"] = not any(f.startswith(("SUSPICIOUSLY_HIGH", "NO_BETTER", "CI_INCLUDES"))
                                                      for f in entry["quality_flags"])
        final = (regressors(0) if meta["task"] == "regression" else classifiers(0))[best]
        final = clone(final).fit(X, y)
        joblib.dump(final, s.models_dir / f"{name}.joblib")
        (s.models_dir / f"{name}.json").write_text(json.dumps({
            "dataset": name, "model": best, "features": list(X.columns), "task": meta["task"], "label": meta["label"],
            "trained_utc": report["generated_utc"], "n_participants": int(n_groups), "disclaimer": DISCLAIMER,
            "cv_summary": res[best], "quality_flags": entry["quality_flags"],
            "usable_for_research_scores": entry["usable_for_research_scores"]}, indent=2))
        report["datasets"][name] = entry
        lvl = res[best].get("participant_level") or res[best]["recording_level"]
        print(f"[train] {name}: best={best} " + (f"MAE={lvl['mae']:.2f}" if meta['task'] == 'regression'
                                                 else f"AUC={lvl.get('roc_auc', float('nan')):.3f}") +
              f" ({entry['seconds']}s)")
    (s.reports_dir / "training_report.json").write_text(json.dumps(report, indent=2))
    (s.reports_dir / "training_report.md").write_text(to_markdown(report))
    print(f"[train] wrote {s.reports_dir / 'training_report.md'}")


def _fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def to_markdown(rep):
    L = ["# ParkiScanML — training report", "", f"Generated {rep['generated_utc']}", "", f"> {rep['disclaimer']}", ""]
    for name, d in rep["datasets"].items():
        L += [f"## {name}", ""]
        if d.get("status") != "evaluated":
            L += [d.get("status", ""), ""]
            continue
        L += [f"- Task: {d['task']} — {d['label']}", f"- Unit: {d['unit']}",
              f"- Rows: {d['n_rows']}, participants: {d['n_participants']}" +
              (f" ({d['n_positive_participants']} positive)" if d.get("n_positive_participants") is not None else ""),
              f"- CV: {d['cv']} (participant-grouped)", f"- Best non-baseline model: **{d['best_model']}**",
              f"- Usable for research scores: **{'yes' if d.get('usable_for_research_scores') else 'no'}**"]
        for fl in d.get("quality_flags", []):
            L.append(f"- ⚠ {fl}")
        if d.get("note"):
            L.append(f"- Note: {d['note']}")
        L.append("")
        if d["task"] == "regression":
            L += ["| model | MAE | R² |", "|---|---|---|"]
            for m, r in d["results"].items():
                L.append(f"| {m} | {_fmt(r['recording_level']['mae'], 2)} | {_fmt(r['recording_level']['r2'])} |")
        else:
            lvl = "participant_level" if any("participant_level" in r for r in d["results"].values()) else "recording_level"
            L += [f"Metrics at **{lvl.replace('_', ' ')}** (pooled out-of-fold):", "",
                  "| model | ROC AUC | 95% CI (participant bootstrap) | AUC mean±sd over repeats | balanced acc | sens | spec |",
                  "|---|---|---|---|---|---|---|"]
            for m, r in d["results"].items():
                x = r.get(lvl) or r["recording_level"]
                ci = x.get("roc_auc_95ci_participant_bootstrap") or [None, None]
                rr = r.get("roc_auc_across_repeats") or {}
                L.append(f"| {m} | {_fmt(x.get('roc_auc'))} | {_fmt(ci[0])}–{_fmt(ci[1])} | "
                         f"{_fmt(rr.get('mean'))}±{_fmt(rr.get('sd'))} | {_fmt(x['balanced_accuracy'])} | "
                         f"{_fmt(x['sensitivity'])} | {_fmt(x['specificity'])} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
