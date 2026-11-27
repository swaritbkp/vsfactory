"""The stateless HTTP verifier on an ephemeral loopback port, torn down after each test."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from knotwork.posttrain import server, vcot
from posttrain_fixtures import FIXTURES, task


@pytest.fixture
def base_url() -> Iterator[str]:
    srv, url = server.serve_in_thread("127.0.0.1", 0)
    try:
        yield url
    finally:
        srv.shutdown()
        srv.server_close()


def _call(url: str, body: object | None = None) -> tuple[int, dict]:  # type: ignore[type-arg]
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def _rollout() -> dict:  # type: ignore[type-arg]
    trace = vcot.extract(FIXTURES / "tangle")
    return {"data_source": "knotwork/tangle", "solution_str": trace.formatted_cot,
            "ground_truth": task("tangle")["answer"],
            "extra_info": {"vcot_boxes_1000": trace.boxes_1000, "domain": "tangle"}}  # fmt: skip


def test_health_reports_the_kernel_hash(base_url: str) -> None:
    code, body = _call(base_url + "/health")
    assert code == 200 and body["status"] == "ok" and len(body["kernel_sha256"]) == 64
    assert body["weights"] == {"outcome": 0.60, "grounding": 0.25, "format": 0.15}


def test_score_and_score_batch(base_url: str) -> None:
    good = _rollout()
    code, one = _call(base_url + "/score", good)
    assert code == 200 and one["score"] >= 0.99
    bad = {**good, "solution_str": "\\boxed{none}"}
    code, many = _call(base_url + "/score_batch", {"items": [good, bad]})
    assert code == 200 and many["n"] == 2
    assert many["results"][0] == one and many["results"][1]["score"] < 0.25
    assert many["mean_score"] == pytest.approx((one["score"] + many["results"][1]["score"]) / 2)


def test_bad_requests_get_json_errors(base_url: str) -> None:
    assert _call(base_url + "/score", {"solution_str": "x"})[0] == 400
    assert _call(base_url + "/score_batch", {"items": 3})[0] == 400
    assert _call(base_url + "/nope")[0] == 404
    code, body = server.handle("POST", "/score", b"{not json", "k")
    assert code == 400 and "error" in body
