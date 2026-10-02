"""M7. Recommendation engine: rules.yaml + metrics -> ranked, numbered fixes with real numbers.

Conditions run in a small whitelisted expression language (no eval of arbitrary code, no attribute access).
A rule whose inputs are missing or whose expression is not evaluable simply does not fire, so a failed
analyzer degrades the advice instead of breaking the request.
"""
from __future__ import annotations

import ast
import logging
from typing import Any

import numpy as np

log = logging.getLogger("rules")

SEVERITY_WEIGHT = {"high": 3.0, "medium": 2.0, "low": 1.0}
_FUNCS = {"min": min, "max": max, "abs": abs, "round": round, "len": len}
_ALLOWED = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Compare,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.IfExp,
    ast.Name, ast.Load, ast.Constant, ast.Call,
)


class MissingValue(Exception):
    """An expression referenced a metric that was not produced (analyzer unavailable)."""


_compiled: dict[str, tuple[Any, frozenset[str]]] = {}


def _compile(expr: str):
    if expr in _compiled:
        return _compiled[expr]
    tree = ast.parse(expr.strip(), mode="eval")
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"disallowed syntax in rule expression {expr!r}: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS or node.keywords:
                raise ValueError(f"only {sorted(_FUNCS)} may be called in {expr!r}")
        elif isinstance(node, ast.Name) and node.id not in _FUNCS:
            names.add(node.id)
    _compiled[expr] = (compile(tree, "<rule>", "eval"), frozenset(names))
    return _compiled[expr]


def safe_eval(expr: str | int | float, namespace: dict[str, Any]) -> Any:
    if isinstance(expr, (int, float)):
        return expr
    code, names = _compile(expr)
    missing = names - namespace.keys()
    if missing:
        raise MissingValue(", ".join(sorted(missing)))
    return eval(code, {"__builtins__": {}, **_FUNCS}, namespace)  # noqa: S307 - AST validated above


# ---------- namespace ----------

def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


def flatten_metrics(metrics: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for group, vals in metrics.items():
        if group == "scores_detail" or not isinstance(vals, dict):
            continue
        for k, v in vals.items():
            if _is_num(v):
                out[f"{group}_{k}"] = float(v)
    return out


def focal_description(metrics: dict, regions: list[dict], min_strength: float = 0.5, limit: int = 3) -> str:
    """Names what each strong attention peak sits on, e.g. 'the face, the text "JUNGLE", the background'."""
    peaks = [p for p in (metrics.get("composition") or {}).get("peaks", []) if p.get("strength", 0) >= min_strength]
    order = {"face": 0, "text": 1, "subject": 2}
    labels: list[str] = []
    for p in peaks:
        label = "the background"
        for r in sorted((r for r in regions if r.get("box") and r["type"] in order), key=lambda r: order[r["type"]]):
            x0, y0, x1, y1 = r["box"]
            if x0 <= p["x"] <= x1 and y0 <= p["y"] <= y1:
                label = {"face": "the face", "subject": "the main subject"}.get(r["type"]) or f'the text "{str(r.get("text", ""))[:18]}"'
                break
        labels.append(label)
    if not labels:
        return "several places"
    shown: list[str] = []
    for lab in dict.fromkeys(labels):  # unique, in strength order
        n = labels.count(lab)
        shown.append(f"{lab} ({n} spots)" if n > 1 else lab)
    text = ", ".join(shown[:limit])
    return text + (f" and {len(shown) - limit} more" if len(shown) > limit else "")


def item_namespace(r: dict) -> dict[str, Any]:
    ns = {f"item_{k}": r.get(k) for k in ("type", "box", "attention_pct", "text", "mobile_readable", "contrast",
                                           "height_pct", "emotion", "intensity", "confidence", "subject_overlap")}
    x0, y0, x1, y1 = r["box"]
    ns["item_area"] = max(0, (x1 - x0) * (y1 - y0))
    for k in ("attention_pct", "subject_overlap"):
        ns[f"item_{k}"] = r.get(k) if r.get(k) is not None else 0.0
    return ns


def annotate_regions(regions: list[dict], results: dict) -> None:
    """Adds `subject_overlap` (0..1) to text regions: how much of the line sits inside the main subject.
    Small text inside a photographed device or product label is part of the scene, not a design choice."""
    subject = results.get("subject")
    union = None
    if subject is not None:
        for sr in subject.regions:
            if sr.mask is not None:
                union = sr.mask if union is None else (union | sr.mask)
    for r in regions:
        if r.get("type") != "text" or not r.get("box"):
            continue
        if union is None:
            r["subject_overlap"] = 0.0
            continue
        x0, y0, x1, y1 = r["box"]
        area = max(1, (x1 - x0) * (y1 - y0))
        r["subject_overlap"] = round(float(union[y0:y1, x0:x1].sum()) / area, 3)


# ---------- validation ----------

def validate_rules(cfg: dict) -> None:
    """Fail loudly at startup if rules.yaml has a syntax problem, a duplicate id or a bad severity."""
    seen: set[str] = set()
    for rule in cfg["rules"]:
        rid = rule.get("id")
        if not rid or rid in seen:
            raise ValueError(f"rules.yaml: missing or duplicate rule id {rid!r}")
        seen.add(rid)
        if "message" not in rule or "severity" not in rule:
            raise ValueError(f"rule {rid}: needs `severity` and `message`")
        exprs = [rule.get("when"), rule.get("impact"), rule["severity"] if rule["severity"] not in SEVERITY_WEIGHT else None]
        spec = rule.get("item")
        if spec:
            if not {"from", "where", "by"} <= spec.keys():
                raise ValueError(f"rule {rid}: item needs from, where and by")
            exprs += [spec["where"], spec["by"]]
        for e in exprs:
            if isinstance(e, str):
                _, names = _compile(e)  # raises ValueError on disallowed syntax, SyntaxError on typos
                if names & SEVERITY_WEIGHT.keys():
                    raise ValueError(f"rule {rid}: {sorted(names & SEVERITY_WEIGHT.keys())} used as a bare name; quote the string in {e!r}")


# ---------- evaluation ----------

def _fallbacks(details: dict, overall_weights: dict, need: int, taken: set[str]) -> list[dict]:
    """When too few rules fire (a strong thumbnail), turn the biggest remaining score drag into a concrete tip."""
    cands, strengths = [], []
    for name, ow in overall_weights.items():
        d = details.get(name) or {}
        if not d.get("applicable", True) or d.get("value") is None:
            continue
        for c in d.get("components", []):
            if c.get("score") is None:
                continue
            if c["score"] < 1.0:
                cands.append(((1 - c["score"]) * c["weight"] * ow, name, c))
            strengths.append((c["score"] * c["weight"] * ow, name, c))
    out = []
    for gain, name, c in sorted(cands, key=lambda t: -t[0]):
        rid = f"improve_{name}_{c['id']}"
        if rid in taken:
            continue
        goal = f"; target {c['goal']}" if c.get("goal") else ""
        msg = (f"Small gain available in {name.replace('_', ' ')}: {c['label'][0].lower() + c['label'][1:]} "
               f"(measured {c['value']:g}{goal}).")
        out.append({"id": rid, "severity": "low", "message": msg, "impact": round(min(1.0, gain * 4), 3), "box": None})
        if len(out) >= need:
            break
    if len(out) < need:  # almost nothing left to improve: say what is working so the list is still useful
        for _, name, c in sorted(strengths, key=lambda t: -t[0]):
            rid = f"keep_{name}_{c['id']}"
            if rid in taken or any(o["id"] == rid for o in out):
                continue
            msg = f"Working well, keep it as you iterate: {c['label'][0].lower() + c['label'][1:]} (measured {c['value']:g})."
            out.append({"id": rid, "severity": "low", "message": msg, "impact": 0.02, "box": None})
            if len(out) >= need:
                break
    return out


def evaluate(cfg: dict, metrics: dict, scores: dict | None, regions: list[dict], details: dict | None = None,
             overall_weights: dict | None = None) -> list[dict]:
    base: dict[str, Any] = {**cfg.get("constants", {}), **flatten_metrics(metrics)}
    for k, v in (scores or {}).items():
        if v is not None:
            base[f"scores_{k}"] = float(v)
    base["focal_labels"] = focal_description(metrics, regions)

    fired: list[dict] = []
    for rule in cfg["rules"]:
        try:
            ns = dict(base)
            box = None
            spec = rule.get("item")
            if spec:
                cands = []
                for r in regions:
                    if r.get("type") != spec["from"] or not r.get("box"):
                        continue
                    cns = {**base, **item_namespace(r)}
                    try:
                        if safe_eval(spec["where"], cns):
                            cands.append((safe_eval(spec["by"], cns), r))
                    except (TypeError, ZeroDivisionError):
                        continue
                if not cands:
                    continue
                _, chosen = (max if spec.get("pick", "max") == "max" else min)(cands, key=lambda t: t[0])
                ns.update(item_namespace(chosen))
                box = [int(v) for v in chosen["box"]]
            if rule.get("when") and not safe_eval(rule["when"], ns):
                continue
            sev = rule["severity"]
            if sev not in SEVERITY_WEIGHT:
                sev = safe_eval(sev, ns)
            if sev not in SEVERITY_WEIGHT:
                raise ValueError(f"rule {rule['id']}: severity evaluated to {sev!r}")
            impact = float(np.clip(safe_eval(rule.get("impact", 0.5), ns), 0.0, 1.0))
            message = rule["message"].format_map(ns)
        except (MissingValue, KeyError, TypeError, ZeroDivisionError) as e:
            log.debug("rule %s skipped: %s", rule.get("id"), e)
            continue
        except ValueError as e:  # a broken rule must never fail a user's request (validate_rules catches it at startup)
            log.warning("rule %s is invalid and was skipped: %s", rule.get("id"), e)
            continue
        fired.append({"id": rule["id"], "severity": sev, "message": message, "impact": round(impact, 3), "box": box})

    fired.sort(key=lambda r: -SEVERITY_WEIGHT[r["severity"]] * r["impact"])
    fired = fired[: cfg.get("max_recommendations", 5)]
    need = cfg.get("min_recommendations", 2) - len(fired)
    if need > 0 and details and overall_weights:
        fired += _fallbacks(details, overall_weights, need, {r["id"] for r in fired})
    return fired
