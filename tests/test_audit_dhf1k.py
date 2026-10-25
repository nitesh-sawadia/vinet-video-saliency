"""Tests for the DHF1K disagreement audit (scripts/error_analysis).

No live model, Vertex, or gateway calls: the network is stubbed and a tiny
stand-in network replaces the DHF1K checkpoint.

Run with: .venv/bin/python3 -m unittest tests.test_audit_dhf1k -v
"""

import hashlib
import io
import json
import os
import random
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

import numpy as np
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EA = os.path.join(_REPO_ROOT, "scripts", "error_analysis")
for _p in (os.path.join(_REPO_ROOT, "scripts"), _EA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import audit_dhf1k as ad  # noqa: E402
import review_server as rs  # noqa: E402
import score as sc  # noqa: E402
import vlm  # noqa: E402

_W, _H = 96, 56
_N_VAL = 6


def _blob(cx, cy, sigma=0.08, w=_W, h=_H):
    ys, xs = np.mgrid[0:h, 0:w]
    d2 = ((xs / (w - 1) - cx) ** 2) + ((ys / (h - 1) - cy) ** 2)
    return np.exp(-d2 / (2 * sigma * sigma))


def _write_gray(path, arr):
    Image.fromarray(np.round(np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


def _tree_digest(root):
    digest = hashlib.sha256()
    for base, dirs, files in sorted(os.walk(root)):
        dirs.sort()
        for name in sorted(files):
            path = os.path.join(base, name)
            digest.update(os.path.relpath(path, root).encode())
            with open(path, "rb") as handle:
                digest.update(handle.read())
    return digest.hexdigest()


def build_fixture_split(root, n_val=_N_VAL, seed=0):
    """split_stride8 with n_val val windows + 1 train window, 32 frames/maps each."""
    rng = np.random.default_rng(seed)
    split_root = os.path.join(root, "vinet_outplay", "split_stride8")
    windows = []
    for i in range(n_val + 1):
        split = "val" if i < n_val else "train"
        rel = os.path.join(split, "soccer", "game%d" % (i % 2), "w%03d" % i)
        wdir = os.path.join(split_root, rel)
        os.makedirs(os.path.join(wdir, "images"))
        os.makedirs(os.path.join(wdir, "maps"))
        cx, cy = 0.15 + 0.12 * i, 0.3 + 0.07 * i
        for t in range(1, 33):
            frame = (rng.random((_H, _W, 3)) * 80).astype(np.uint8)
            frame[..., 0] = np.clip(frame[..., 0] + _blob(cx, cy) * 170, 0, 255).astype(np.uint8)
            Image.fromarray(frame).save(os.path.join(wdir, "images", "%04d.png" % t))
            _write_gray(os.path.join(wdir, "maps", "%04d.png" % t), _blob(1 - cx, cy))
        box = [max(0, cx - 0.1), max(0, cy - 0.15), min(1, cx + 0.1), min(1, cy + 0.15)]
        native = [
            {"targets": [{"label": "person", "bbox_xyxy_norm": box, "weight": 1.0}]}
            for _ in range(32)
        ]
        with open(os.path.join(wdir, "boxes.json"), "w") as handle:
            json.dump({"native_frames": native}, handle)
        windows.append(
            {"split": split, "sport": "soccer", "game": "game%d" % (i % 2),
             "dir": "outputs/saliency_vinet/w%03d" % i}
        )
    with open(os.path.join(split_root, "split_manifest.json"), "w") as handle:
        json.dump({"windows": windows}, handle)
    return split_root


class _FakeDHF1K(torch.nn.Module):
    """Deterministic stand-in: saliency = sigmoid of last-frame red channel."""

    def forward(self, x):  # x: B, C, T, H, W
        return torch.sigmoid(x[:, 0, -1] * 2.0).unsqueeze(1)


def _score(split_root, out_dir):
    ckpt = os.path.join(os.path.dirname(out_dir), "fake_dhf1k.pt")
    if not os.path.isfile(ckpt):
        with open(ckpt, "wb") as handle:
            handle.write(b"fake")
    return sc.score_dhf1k_baseline(
        ckpt, split_root, out_dir, batch_size=4, num_workers=0, device_name="cpu",
        model=_FakeDHF1K(),
    )


class _ScriptedTransport:
    """Returns queued (parsed, raw) responses; records every call's content."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, content, *, model, timeout, temperature):
        self.calls.append(content)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item, json.dumps(item)


def _verdict(disposition="gt_likely_wrong", confidence=0.8):
    return {
        "disposition": disposition,
        "confidence": confidence,
        "evidence": {
            "viewer_focus": "the striker shooting at goal",
            "sol_boxes_cover_subject": "no",
            "sol_box_notes": "box sits on an empty patch of grass",
            "dhf1k_peak_lands_on": "salient_subject_or_action",
            "disagreement_cause": "annotation boxes the wrong region",
        },
    }


def _correction(issues=True):
    return {
        "issues_found": issues,
        "notes": "Box moved onto the striker." if issues else "Current annotations are accurate",
        "detections": [
            {"label": "person", "bbox_xyxy_norm": [0.6, 0.2, 0.8, 0.7], "weight": 0.7,
             "confidence": 0.9},
            {"label": "ball", "bbox_xyxy_norm": [0.55, 0.6, 0.6, 0.66], "weight": 0.3,
             "confidence": 0.8},
        ],
    }


class _FixtureCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dhf1k_audit_test_")
        self.split_root = build_fixture_split(self.tmp)
        self.out_dir = os.path.join(self.tmp, "audit")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class LastFrameRankingTest(_FixtureCase):
    def test_scorer_opens_only_maps_0032(self):
        import outplay_dataset as od

        opened = []
        real = od._load_map

        def spy(path):
            opened.append(os.path.basename(path))
            return real(path)

        with mock.patch.object(od, "_load_map", side_effect=spy):
            _score(self.split_root, self.out_dir)
        self.assertEqual(len(opened), _N_VAL)
        self.assertEqual(set(opened), {"0032.png"})

    def test_ranking_ignores_other_31_maps(self):
        _score(self.split_root, self.out_dir)
        ad.select_candidates(self.out_dir, top_percent=50, min_candidates=2, max_candidates=10)
        first = ad._read_jsonl(os.path.join(self.out_dir, "ranking.jsonl"))

        rng = np.random.default_rng(123)
        for base, _dirs, files in os.walk(self.split_root):
            if os.path.basename(base) != "maps":
                continue
            for name in files:
                if name != "0032.png":
                    _write_gray(os.path.join(base, name), rng.random((_H, _W)))
        out2 = os.path.join(self.tmp, "audit2")
        _score(self.split_root, out2)
        ad.select_candidates(out2, top_percent=50, min_candidates=2, max_candidates=10)
        second = ad._read_jsonl(os.path.join(out2, "ranking.jsonl"))

        key = lambda rows: [(r["id"], r["rank"], r["disagreement_score"], r["components"])
                            for r in rows]
        self.assertEqual(key(first), key(second))

    def test_last_map_change_changes_score(self):
        _score(self.split_root, self.out_dir)
        before = {r["id"]: r["components"]["cc"]
                  for r in ad._read_jsonl(os.path.join(self.out_dir, "dhf1k_scores.jsonl"))}
        target = os.path.join(self.split_root, "val", "soccer", "game0", "w000", "maps", "0032.png")
        _write_gray(target, _blob(0.15, 0.3))
        out2 = os.path.join(self.tmp, "audit2")
        _score(self.split_root, out2)
        after = {r["id"]: r["components"]["cc"]
                 for r in ad._read_jsonl(os.path.join(out2, "dhf1k_scores.jsonl"))}
        self.assertNotAlmostEqual(before["00000_w000"], after["00000_w000"], places=3)

    def test_rank_is_deterministic_with_caps_and_exclusions(self):
        rows = []
        for i in range(40):
            rows.append({
                "id": "%05d_w" % i,
                "components": {"kldiv": float(i % 7), "cc": 1.0 - i / 40.0, "sim": 0.5,
                               "peak_distance_norm": i / 40.0, "shape_mismatch": (i * 3 % 11) / 11.0,
                               "gt_mass": 10.0, "pred_range": 1.0},
            })
        rows.append({"id": "nan", "components": {**rows[0]["components"], "cc": float("nan")}})
        rows.append({"id": "empty", "components": {**rows[0]["components"], "gt_mass": 0.0}})
        ranked, sel = ad.rank_disagreements(rows, top_percent=10, min_candidates=6, max_candidates=8)
        shuffled = rows[:]
        random.Random(7).shuffle(shuffled)
        ranked2, sel2 = ad.rank_disagreements(shuffled, top_percent=10, min_candidates=6,
                                              max_candidates=8)
        self.assertEqual([r["id"] for r in ranked], [r["id"] for r in ranked2])
        self.assertEqual(sel, sel2)
        self.assertEqual(sel["n_eligible"], 40)
        self.assertEqual(sel["n_top_percent"], 4)
        self.assertEqual(sel["n_candidates"], 6)
        cands = [r for r in ranked if r["candidate"]]
        self.assertEqual(len(cands), 6)
        self.assertEqual([r["selection_reason"] for r in cands],
                         ["top_percent"] * 4 + ["min_candidates"] * 2)
        self.assertEqual(sel["candidate_threshold"], cands[-1]["disagreement_score"])
        excluded = {r["id"]: r["exclusion_reason"] for r in ranked if not r["eligible"]}
        self.assertEqual(excluded, {"nan": "nonfinite_cc", "empty": "empty_gt"})
        scores = [r["disagreement_score"] for r in ranked if r["eligible"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_max_cap_applies(self):
        rows = [{"id": str(i), "components": {"kldiv": i, "cc": -i, "sim": -i,
                 "peak_distance_norm": i, "shape_mismatch": i, "gt_mass": 1, "pred_range": 1}}
                for i in range(100)]
        ranked, sel = ad.rank_disagreements(rows, top_percent=20, min_candidates=5, max_candidates=10)
        self.assertEqual(sel["n_candidates"], 10)
        self.assertEqual(ranked[0]["id"], "99")
        self.assertEqual(ranked[10]["selection_reason"], "max_candidates_cap")
        self.assertEqual(ranked[25]["selection_reason"], "below_threshold")

    def test_path_guards(self):
        with self.assertRaises(SystemExit):
            ad.check_audit_paths(os.path.join(self.tmp, "split"), self.out_dir)
        with self.assertRaises(SystemExit):
            ad.check_audit_paths(self.split_root, os.path.join(self.split_root, "audit"))
        with self.assertRaises(SystemExit):
            ad.check_audit_paths(self.split_root, os.path.join(os.path.dirname(self.split_root), "a"))
        ad.check_audit_paths(self.split_root, self.out_dir)


class VLMSchemaTest(unittest.TestCase):
    def test_valid_verdict(self):
        out = vlm.validate_dhf1k_verdict(_verdict())
        self.assertEqual(out["disposition"], "gt_likely_wrong")
        self.assertEqual(set(out), {"disposition", "confidence", "evidence"})

    def test_verdict_rejections(self):
        bad = [
            {**_verdict(), "disposition": "model_wrong"},
            {**_verdict(), "confidence": 1.4},
            {**_verdict(), "confidence": "high"},
            {**_verdict(), "evidence": {**_verdict()["evidence"], "viewer_focus": ""}},
            {**_verdict(), "evidence": {**_verdict()["evidence"], "dhf1k_peak_lands_on": "crowd"}},
            {**_verdict(), "evidence": {**_verdict()["evidence"], "sol_boxes_cover_subject": "maybe"}},
            {**_verdict(), "detections": []},
            {k: v for k, v in _verdict().items() if k != "evidence"},
        ]
        for item in bad:
            with self.assertRaises(vlm.VLMValidationError, msg=str(item)):
                vlm.validate_dhf1k_verdict(item)

    def test_valid_correction(self):
        out = vlm.validate_sol_correction(_correction())
        self.assertTrue(out["issues_found"])
        self.assertEqual(len(out["detections"]), 2)

    def test_correction_rejections(self):
        base = _correction()
        det = base["detections"][0]

        def with_det(**kw):
            return {**base, "detections": [{**det, **kw, "weight": 1.0}]}

        bad = [
            with_det(bbox_xyxy_norm=[120, 40, 300, 200]),
            with_det(bbox_xyxy_norm=[0.5, 0.5, 0.4, 0.8]),
            with_det(bbox_xyxy_norm=[0.0, 0.0, 1.0, 1.0]),
            with_det(bbox_xyxy_norm=[0.1, 0.2, 0.3]),
            with_det(label="player"),
            with_det(confidence=2.0),
            {**base, "detections": [{**det, "weight": 0.5}]},
            {**base, "detections": []},
            {**base, "issues_found": "yes"},
            {**base, "notes": ""},
        ]
        for item in bad:
            with self.assertRaises(vlm.VLMValidationError, msg=json.dumps(item)):
                vlm.validate_sol_correction(item)

    def test_no_issues_may_return_empty_list(self):
        out = vlm.validate_sol_correction({"issues_found": False,
                                           "notes": "Current annotations are accurate",
                                           "detections": []})
        self.assertFalse(out["issues_found"])

    def _sol_image(self):
        return Image.new("RGB", (_W, _H))

    def test_correction_retries_once_with_validator_error(self):
        bad = {**_correction(), "detections": [{**_correction()["detections"][0], "weight": 0.2}]}
        transport = _ScriptedTransport([bad, _correction()])
        out = vlm.correct_sol_boxes({"sport": "soccer"}, sol_image=self._sol_image(),
                                    sol_targets=[], transport=transport)
        self.assertEqual(out["status"], "valid")
        self.assertEqual(out["attempts"], 2)
        retry_text = transport.calls[1][-1]["text"]
        self.assertIn("weights must sum to 1.0", retry_text)

    def test_correction_invalid_after_retry(self):
        bad = {**_correction(), "detections": [{**_correction()["detections"][0], "label": "dog",
                                                "weight": 1.0}]}
        transport = _ScriptedTransport([bad, bad])
        out = vlm.correct_sol_boxes({"sport": "soccer"}, sol_image=self._sol_image(),
                                    sol_targets=[], transport=transport)
        self.assertEqual(out["status"], "invalid")
        self.assertEqual(out["detections"], [])
        self.assertEqual(len(transport.calls), 2)

    def test_verdict_prompt_has_three_images_and_no_boxes_request(self):
        img = self._sol_image()
        content = vlm.build_dhf1k_verdict_content({"sport": "soccer"}, sol_image=img,
                                                  gt_overlay=img, pred_overlay=img, sol_targets=[])
        self.assertEqual(sum(1 for p in content if p["type"] == "image_url"), 3)
        text = content[0]["text"].lower()
        self.assertIn("do not draw or return boxes", text)
        self.assertNotIn("match the model", text)
        self.assertNotIn("match the prediction", text)

    def test_correction_prompt_has_only_the_sol_frame(self):
        content = vlm.build_sol_correction_content({"sport": "soccer"}, sol_image=self._sol_image(),
                                                   sol_targets=[])
        self.assertEqual(sum(1 for p in content if p["type"] == "image_url"), 1)
        text = content[0]["text"]
        self.assertIn("issues_found", text)
        self.assertIn("%d x %d pixels" % (_W, _H), text)
        self.assertNotIn("match the model", text.lower())


class NoCorrectionOutcomesTest(_FixtureCase):
    def setUp(self):
        super().setUp()
        _score(self.split_root, self.out_dir)
        ad.select_candidates(self.out_dir, top_percent=50, min_candidates=4, max_candidates=4)
        cands = ad._read_jsonl(os.path.join(self.out_dir, "candidates.jsonl"))
        self.ids = [c["id"] for c in sorted(cands, key=lambda c: c["rank"])]
        self.assertEqual(len(self.ids), 4)

    def _results(self):
        return {r["id"]: r for r in ad._read_jsonl(os.path.join(self.out_dir, "vlm_results.jsonl"))}

    def test_outcomes(self):
        bad_corr = {**_correction(), "detections": []}
        transport = _ScriptedTransport([
            _verdict("gt_correct_model_wrong"),
            _verdict("gt_likely_wrong"), _correction(issues=False),
            _verdict("gt_likely_wrong"), bad_corr, bad_corr,
            _verdict("gt_likely_wrong"), _correction(),
        ])
        ad.adjudicate_candidates(self.out_dir, transport=transport)
        self.assertEqual(transport.responses, [])
        res = self._results()
        r0, r1, r2, r3 = (res[i] for i in self.ids)
        self.assertEqual(r0["correction"]["status"], "not_requested")
        self.assertEqual(r1["correction"]["status"], "no_issues")
        self.assertEqual(r2["correction"]["status"], "invalid")
        self.assertEqual(r2["verdict"]["disposition"], "gt_likely_wrong")
        self.assertEqual(r3["correction"]["status"], "valid")
        self.assertEqual([r["accept_allowed"] for r in (r0, r1, r2, r3)],
                         [False, False, False, True])
        self.assertIsNone(r0["proposed_media"])
        self.assertTrue(os.path.isfile(os.path.join(self.out_dir, r3["proposed_media"])))

        index = ad.build_review_index(self.out_dir)
        by_id = {s["id"]: s for s in index["samples"]}
        self.assertEqual(index["kind"], "dhf1k_audit")
        self.assertTrue(by_id[self.ids[3]]["accept_allowed"])
        self.assertNotIn("raw", by_id[self.ids[3]]["verdict"])

    def test_transport_error_and_resume(self):
        transport = _ScriptedTransport([RuntimeError("Vertex HTTP 503"),
                                        _verdict("insufficient_evidence")])
        ad.adjudicate_candidates(self.out_dir, transport=transport, ids=[self.ids[0]])
        self.assertEqual(self._results()[self.ids[0]]["verdict"]["status"], "error")
        ad.adjudicate_candidates(self.out_dir, transport=transport, ids=[self.ids[0]])
        self.assertEqual(self._results()[self.ids[0]]["verdict"]["disposition"],
                         "insufficient_evidence")
        ad.adjudicate_candidates(self.out_dir, transport=_ScriptedTransport([]), ids=[self.ids[0]])


class ReviewServerTest(_FixtureCase):
    def setUp(self):
        super().setUp()
        self.split_digest = _tree_digest(self.split_root)
        _score(self.split_root, self.out_dir)
        ad.select_candidates(self.out_dir, top_percent=50, min_candidates=2, max_candidates=2)
        self.ids = [c["id"] for c in sorted(ad._read_jsonl(
            os.path.join(self.out_dir, "candidates.jsonl")), key=lambda c: c["rank"])]
        transport = _ScriptedTransport([_verdict("gt_likely_wrong"), _correction(),
                                        _verdict("both_plausible_or_ambiguous")])
        ad.adjudicate_candidates(self.out_dir, transport=transport)
        ad.build_review_index(self.out_dir)
        self.server = rs.make_review_server(self.out_dir, port=0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        super().tearDown()

    def _url(self, path):
        return "http://127.0.0.1:%d%s" % (self.port, path)

    def _post(self, payload):
        req = urllib.request.Request(self._url("/api/review"), data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def _get_json(self, path):
        with urllib.request.urlopen(self._url(path), timeout=10) as resp:
            return json.loads(resp.read())

    def test_accept_writes_human_reviews_and_leaves_split_untouched(self):
        accept_id, ambiguous_id = self.ids
        self.assertEqual(self._get_json("/api/reviews"), {})
        status, record = self._post({"id": accept_id, "decision": "accept",
                                     "accepted_boxes": [{"label": "ball"}]})
        self.assertEqual(status, 200)
        expected = vlm.validate_sol_correction(_correction())["detections"]
        self.assertEqual(record["accepted_boxes"], expected)

        with open(os.path.join(self.out_dir, "human_reviews.json")) as handle:
            saved = json.load(handle)
        self.assertEqual(saved[accept_id]["decision"], "accept")
        self.assertEqual(saved[accept_id]["accepted_boxes"], expected)
        self.assertTrue(saved[accept_id]["timestamp"])
        self.assertEqual(self._get_json("/api/reviews")[accept_id]["decision"], "accept")

        status, body = self._post({"id": ambiguous_id, "decision": "accept"})
        self.assertEqual(status, 400)
        self.assertIn("gt_likely_wrong", body["error"])
        self.assertEqual(self._post({"id": ambiguous_id, "decision": "skip"})[0], 200)
        self.assertEqual(self._post({"id": "nope", "decision": "reject"})[0], 400)
        self.assertEqual(self._post({"id": accept_id, "decision": "maybe"})[0], 400)

        exported = ad.accepted_corrections(self.out_dir)
        self.assertEqual([r["id"] for r in exported], [accept_id])

        self.assertEqual(self._post({"id": accept_id, "decision": "reject"})[0], 200)
        saved = self._get_json("/api/reviews")
        self.assertEqual(saved[accept_id]["decision"], "reject")
        self.assertNotIn("accepted_boxes", saved[accept_id])
        self.assertEqual(ad.accepted_corrections(self.out_dir), [])

        self.assertEqual(_tree_digest(self.split_root), self.split_digest)

    def test_ui_and_media_are_served(self):
        with urllib.request.urlopen(self._url("/"), timeout=10) as resp:
            html = resp.read().decode()
        self.assertIn("bootAudit", html)
        self.assertIn("/api/review", html)
        sample = self._get_json("/review_index.json")["samples"][0]
        for rel in sample["media"].values():
            with urllib.request.urlopen(self._url("/" + rel), timeout=10) as resp:
                self.assertEqual(resp.status, 200)


class VertexTransportTest(unittest.TestCase):
    def setUp(self):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = self.key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode()
        self.info = {"client_email": "sa@test-proj.iam.gserviceaccount.com", "private_key": pem,
                     "private_key_id": "kid1", "project_id": "test-proj",
                     "token_uri": "https://oauth2.googleapis.com/token"}
        vlm._TOKEN_CACHE.clear()

    def test_jwt_is_rs256_signed(self):
        import base64

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        token = vlm.build_service_account_jwt(self.info, now=1000)
        head_b64, claims_b64, sig_b64 = token.split(".")
        pad = lambda s: s + "=" * (-len(s) % 4)
        header = json.loads(base64.urlsafe_b64decode(pad(head_b64)))
        claims = json.loads(base64.urlsafe_b64decode(pad(claims_b64)))
        self.assertEqual(header["alg"], "RS256")
        self.assertEqual(claims["aud"], self.info["token_uri"])
        self.assertEqual(claims["exp"] - claims["iat"], 3600)
        self.key.public_key().verify(base64.urlsafe_b64decode(pad(sig_b64)),
                                     ("%s.%s" % (head_b64, claims_b64)).encode(),
                                     padding.PKCS1v15(), hashes.SHA256())

    def test_vertex_request_and_token_cache(self):
        seen = []

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(request, timeout=None):
            seen.append(request)
            if request.full_url.endswith("/token"):
                return _Resp(json.dumps({"access_token": "tok", "expires_in": 3600}).encode())
            body = {"candidates": [{"content": {"parts": [{"text": json.dumps(_verdict())}]}}]}
            return _Resp(json.dumps(body).encode())

        env = {"VERTEX_AI_SERVICE_ACCOUNT": json.dumps(self.info), "BASE_URL": "http://gw",
               "GEMINI_API_KEY": "x"}
        content = [{"type": "text", "text": "hi"},
                   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}]
        with mock.patch.object(vlm, "load_env", return_value=env), \
                mock.patch.object(vlm.urllib.request, "urlopen", side_effect=fake_urlopen):
            parsed, _raw = vlm.vlm_generate_json(content, model="gemini-3.6-flash")
            vlm.vlm_generate_json(content, model="vertex_ai/gemini-3.6-flash")
        self.assertEqual(parsed["disposition"], "gt_likely_wrong")
        token_calls = [r for r in seen if r.full_url.endswith("/token")]
        gen_calls = [r for r in seen if r.full_url.endswith(":generateContent")]
        self.assertEqual(len(token_calls), 1)
        self.assertEqual(len(gen_calls), 2)
        self.assertEqual(
            gen_calls[0].full_url,
            "https://aiplatform.googleapis.com/v1/projects/test-proj/locations/global/"
            "publishers/google/models/gemini-3.6-flash:generateContent",
        )
        self.assertEqual(gen_calls[0].get_header("Authorization"), "Bearer tok")
        body = json.loads(gen_calls[0].data)
        self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(body["contents"][0]["parts"][1]["inlineData"],
                         {"mimeType": "image/jpeg", "data": "QUJD"})

    def test_gateway_only_without_service_account(self):
        env = {"BASE_URL": "http://gw", "GEMINI_API_KEY": "x"}
        with mock.patch.object(vlm, "load_env", return_value=env), \
                mock.patch.object(vlm, "_chat_completion", return_value=({}, "{}")) as chat:
            vlm.vlm_generate_json([{"type": "text", "text": "hi"}], model="gemini-3.6-flash")
        self.assertEqual(chat.call_args.kwargs["model"], "vertex_ai/gemini-3.6-flash")
        self.assertEqual(chat.call_args.kwargs["base_url"], "http://gw")


if __name__ == "__main__":
    unittest.main()
