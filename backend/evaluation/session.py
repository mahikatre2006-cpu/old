"""M11 service layer: a gaze session in, CC / NSS / ground-truth map out; plus per-thumbnail aggregation."""
from __future__ import annotations

import uuid

import numpy as np

from evaluation.gaze import build_ground_truth, first_look, gaussian_map, score_session, to_map_points
from evaluation.metrics import cc, nss
from fusion.heatmap import decode_attention_png, encode_heatmap_png


class GazeError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _prediction(store, thumbnail_id: str) -> tuple[dict, np.ndarray]:
    analysis = store.get_analysis(thumbnail_id)
    if analysis is None:
        raise GazeError(404, "Unknown thumbnail id. Analyze the thumbnail first.")
    if not analysis.get("attention_png") or not analysis.get("scanpath"):
        raise GazeError(409, "This analysis has no attention map (it predates the lab). Analyze the thumbnail again.")
    return analysis, decode_attention_png(analysis["attention_png"])


def run_gaze_session(store, cfg: dict, body: dict) -> dict:
    analysis, pred = _prediction(store, body["thumbnail_id"])
    points = body["points"]
    if len(points) > cfg["limits"]["max_points"]:
        raise GazeError(422, f"Too many points (max {cfg['limits']['max_points']}).")
    try:
        truth = build_ground_truth(body["method"], points, cfg)
    except ValueError as e:
        raise GazeError(422, str(e)) from e

    scores = score_session(pred, truth, cfg)
    look = first_look(truth["fixations"], analysis["scanpath"], cfg["first_look"]["hit_radius_px"])
    sid = uuid.uuid4().hex
    store.save_gaze_session(
        sid,
        body["thumbnail_id"],
        body["method"],
        {"points": points, "fixations": truth["fixations"], "screen": body.get("screen"), "participant": body.get("participant"), **look, "sim": scores["sim"], "kld": scores["kld"]},
        scores["cc"],
        scores["nss"],
    )
    return {
        "session_id": sid,
        "method": body["method"],
        "cc": scores["cc"],
        "nss": scores["nss"],
        "sim": scores["sim"],
        "kld": scores["kld"],
        "n_points": len(points),
        "n_fixations": len(truth["fixations"]),
        "fixations": truth["fixations"],
        "warnings": truth["warnings"],
        "ground_truth_png": encode_heatmap_png(truth["gt"]),
        **look,
    }


def aggregate(store, cfg: dict, thumbnail_id: str) -> dict:
    """All saved sessions for one thumbnail: per-session metrics, hit rate, and the participants' pooled human map
    scored against the prediction (pooling many viewers is a much steadier ground truth than any one session)."""
    analysis, pred = _prediction(store, thumbnail_id)
    rows = store.list_gaze_sessions(thumbnail_id)
    if not rows:
        return {"thumbnail_id": thumbnail_id, "n_sessions": 0, "sessions": [], "mean_cc": None, "mean_nss": None, "first_look_hit_rate": None,
                "aggregate_cc": None, "aggregate_nss": None, "aggregate_png": None}

    maps, all_pts, all_w = [], [], []
    for r in rows:
        try:
            t = build_ground_truth(r["method"], r["payload"]["points"], cfg)
        except ValueError:
            continue
        maps.append(t["gt"])
        all_pts += t["nss_points"]
        all_w += list(t["weights"]) if t["weights"] is not None else [1.0] * len(t["nss_points"])
    pooled = np.mean(maps, axis=0) if maps else None
    if pooled is not None and pooled.max() > 0:
        pooled = pooled / pooled.max()

    hits = [r["payload"].get("first_look_hit") for r in rows if r["payload"].get("first_look_hit") is not None]
    out = {
        "thumbnail_id": thumbnail_id,
        "n_sessions": len(rows),
        "sessions": [{"id": r["id"], "method": r["method"], "cc": r["cc"], "nss": r["nss"], "participant": r["payload"].get("participant"),
                      "first_look_hit": r["payload"].get("first_look_hit")} for r in rows],
        "mean_cc": round(float(np.mean([r["cc"] for r in rows])), 4),
        "mean_nss": round(float(np.mean([r["nss"] for r in rows])), 4),
        "first_look_hit_rate": round(sum(bool(h) for h in hits) / len(hits), 3) if hits else None,
        "aggregate_cc": None, "aggregate_nss": None, "aggregate_png": None,
    }
    if pooled is not None:
        import cv2

        p = cv2.resize(pred.astype(np.float32), (pooled.shape[1], pooled.shape[0]), interpolation=cv2.INTER_AREA)
        out["aggregate_cc"] = round(cc(p, pooled), 4)
        out["aggregate_nss"] = round(nss(p, to_map_points(all_pts, (pooled.shape[1], pooled.shape[0])), all_w), 4)
        out["aggregate_png"] = encode_heatmap_png(pooled)
    return out
