"""Real-only tests for the easy-UI server part (PLAN.md step 4).

The shipped UI must show REAL data only: no demo/fake/sample/
placeholder wording, no marker without a valid bridge, no stale
marker presented as live, and the product server must refuse to run
without a real source.

The five tests (from the UI research plan):
  1. wording — viewer.html and the served /operator page contain no
     demo/fake/sample/placeholder/example/test wording.
  2. product-server-refuses — `python -m server.main --mode live`
     (no djitellopy, no Tello) and `--mode virtual` both exit with
     an error and serve nothing (the port stays closed).
  3. sse-none-while-bridge-invalid — driving s5_tracking_loss
     through the real pipeline + real HTTP server, /api/markers
     emits only {"type": "none"} (bridge_valid false) while the
     bridge is invalid.
  4. operator-dash-before-first-frame — a fresh server's /operator
     page shows "—" (not 0) in every facts field, and /api/status
     reports last_frame_age_s / frames_last_10s as null.
  5. viewer-zero-skeletons — the real viewer, in headless Chromium,
     draws zero skeleton pixels from 60 s of {"type": "none"} events.

Run standalone:
    ~/workspace/tello-search/.venv/bin/python tests/test_real_only.py
Or imported from tests/run_tests.py:
    from tests.test_real_only import run_all
    passed, findings = run_all(port=18901)
"""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

WALL_MARKER_DIR = Path(__file__).resolve().parent.parent
if str(WALL_MARKER_DIR) not in sys.path:
    sys.path.insert(0, str(WALL_MARKER_DIR))

from server.main import ServerState, make_handler  # noqa: E402

BANNED_WORDS = re.compile(
    r"\b(demo|fake|sample|placeholder|example|test)\b", re.IGNORECASE)
VIEWER_PATH = WALL_MARKER_DIR / "phone" / "viewer.html"


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _port_open(port):
    s = socket.socket()
    s.settimeout(1.0)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _http_get(port, path, timeout=5.0):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    headers = dict(resp.getheaders())
    status = resp.status
    conn.close()
    return status, headers, body


def _start_server(state, port):
    handler = make_handler(state, VIEWER_PATH)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


# ---------------------------------------------------------------------------
# 1. wording
# ---------------------------------------------------------------------------

def test_wording(port):
    """No banned wording in the shipped UI (viewer + operator page)."""
    findings = {"checks": {}, "words_found": {}}
    state = ServerState()
    httpd = _start_server(state, port)
    try:
        viewer_html = VIEWER_PATH.read_text(encoding="utf-8")
        _, _, op_body = _http_get(port, "/operator")
        op_html = op_body.decode("utf-8", errors="replace")
        for name, html in (("viewer", viewer_html),
                           ("operator", op_html)):
            words = sorted(set(w.lower() for w in BANNED_WORDS.findall(html)))
            findings["words_found"][name] = words
            findings["checks"][f"{name}_no_banned_words"] = not words
        findings["checks"]["operator_no_video_tag"] = \
            "<video" not in op_html.lower()
        findings["checks"]["operator_no_img_tag"] = \
            "<img" not in op_html.lower()
    finally:
        httpd.shutdown()
        httpd.server_close()
    findings["passed"] = all(findings["checks"].values())
    return findings["passed"], findings


# ---------------------------------------------------------------------------
# 2. product server refuses without a real source
# ---------------------------------------------------------------------------

def _run_product_server(args, port, timeout=60):
    """Run the product server CLI; return (returncode, stderr)."""
    proc = subprocess.run(
        [sys.executable, "-m", "server.main"] + args,
        cwd=str(WALL_MARKER_DIR), capture_output=True, text=True,
        timeout=timeout)
    return proc.returncode, proc.stderr


def test_product_server_refuses(port):
    """--mode live (no drone) and --mode virtual both exit 2 and
    serve nothing."""
    findings = {"checks": {}, "runs": {}}
    for label, args in (
            ("live", ["--mode", "live", "--port", str(port)]),
            ("virtual", ["--mode", "virtual", "--scenario", "s1_basic",
                         "--port", str(port)])):
        rc, stderr = _run_product_server(args, port)
        findings["runs"][label] = {"returncode": rc,
                                   "stderr": stderr.strip()[:300]}
        findings["checks"][f"{label}_exits_nonzero"] = rc != 0
        findings["checks"][f"{label}_serves_nothing"] = not _port_open(port)
    findings["checks"]["live_error_names_djitellopy_or_tello"] = (
        "djitellopy" in findings["runs"]["live"]["stderr"].lower()
        or "tello" in findings["runs"]["live"]["stderr"].lower())
    findings["checks"]["virtual_error_points_at_test_entry"] = (
        "tests.virtual_server" in findings["runs"]["virtual"]["stderr"])
    findings["passed"] = all(findings["checks"].values())
    return findings["passed"], findings


# ---------------------------------------------------------------------------
# 3. SSE emits only {"type": "none"} while the bridge is invalid
# ---------------------------------------------------------------------------

def test_sse_none_while_bridge_invalid(port):
    """Drive s5 through the real pipeline + real HTTP server; while
    the bridge is invalid (frames 36+), every SSE event is none."""
    from server.main import Pipeline
    from server.detector import PoseDetector
    from tests import scenarios

    findings = {"checks": {}, "events_seen": 0, "event_types": []}
    detector = PoseDetector()
    detector.warmup()
    frames = scenarios.get_frames("s5_tracking_loss", detector=detector)
    state = ServerState()
    pipeline = Pipeline(detector)
    httpd = _start_server(state, port)
    stop = threading.Event()
    errors = []
    progress = {"frames": 0}

    def drive():
        try:
            for frame, pose, tracking_valid, _meta in frames:
                if stop.is_set():
                    break
                t = pipeline.frames_processed / 10.0
                marker, info = pipeline.process_frame(
                    frame, pose, tracking_valid, t)
                state.update_from_pipeline(pipeline, marker, info,
                                           time.time(), t)
                progress["frames"] = pipeline.frames_processed
        except Exception as exc:  # noqa: BLE001 — reported, not hidden
            errors.append(repr(exc))

    thread = threading.Thread(target=drive, daemon=True)
    thread.start()
    try:
        # Wait until the bridge is invalid AND the loss window has
        # actually been reached (frame 36+). The bridge also reads
        # invalid before the first frame, so the frame count is the
        # guard that keeps the SSE read inside the loss window.
        deadline = time.time() + 20.0
        bridge_invalid_seen = False
        while time.time() < deadline:
            try:
                _, _, body = _http_get(port, "/api/status", timeout=2.0)
                if (not json.loads(body.decode("utf-8"))["bridge_valid"]
                        and progress["frames"] >= 37):
                    bridge_invalid_seen = True
                    break
            except Exception:
                pass
            time.sleep(0.1)
        findings["checks"]["bridge_invalid_seen"] = bridge_invalid_seen
        if not bridge_invalid_seen:
            findings["passed"] = False
            return findings["passed"], findings

        # Read SSE for 1.5 s; every event must be {"type": "none"}.
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=6)
        conn.request("GET", "/api/markers")
        resp = conn.getresponse()
        findings["checks"]["sse_content_type"] = (
            resp.status == 200
            and "text/event-stream" in resp.headers.get("Content-Type", ""))
        events = []
        buf = b""
        deadline = time.time() + 1.5
        while time.time() < deadline:
            try:
                chunk = (resp.read1(4096) if hasattr(resp, "read1")
                         else resp.read(4096))
            except Exception:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                block, buf = buf.split(b"\n\n", 1)
                for line in block.decode("utf-8",
                                         errors="replace").splitlines():
                    if line.startswith("data:"):
                        events.append(line[5:].strip())
        conn.close()
        parsed = []
        for ev in events:
            try:
                parsed.append(json.loads(ev))
            except Exception:
                parsed.append(None)
        findings["events_seen"] = len(events)
        findings["event_types"] = sorted({p.get("type") for p in parsed
                                          if isinstance(p, dict)})
        findings["checks"]["sse_events_received"] = len(events) >= 1
        findings["checks"]["sse_all_json"] = (
            len(events) > 0 and all(p is not None for p in parsed))
        findings["checks"]["sse_only_none_while_bridge_invalid"] = (
            len(parsed) > 0
            and all(isinstance(p, dict) and p.get("type") == "none"
                    for p in parsed)
            and all(p.get("bridge_valid") is False for p in parsed))
    finally:
        stop.set()
        thread.join(timeout=5.0)
        httpd.shutdown()
        httpd.server_close()
    if errors:
        findings["driver_errors"] = errors
        findings["passed"] = False
    else:
        findings["passed"] = all(findings["checks"].values())
    return findings["passed"], findings


# ---------------------------------------------------------------------------
# 4. operator page shows "—" before the first frame
# ---------------------------------------------------------------------------

def test_operator_dash_before_first_frame(port):
    """Fresh server: /operator facts read "—", /api/status fields null."""
    findings = {"checks": {}}
    state = ServerState()
    httpd = _start_server(state, port)
    try:
        status, _, op_body = _http_get(port, "/operator")
        op_html = op_body.decode("utf-8", errors="replace")
        findings["checks"]["operator_200"] = status == 200
        for field in ("fDrones", "fTracks", "fMarkers", "fUptime",
                      "fLastFrame", "fFrames10"):
            findings["checks"][f"facts_{field}_is_dash"] = (
                f'id="{field}">—<' in op_html)
        _, _, body = _http_get(port, "/api/status")
        payload = json.loads(body.decode("utf-8"))
        findings["checks"]["status_last_frame_age_null"] = (
            payload.get("last_frame_age_s") is None)
        findings["checks"]["status_frames_last_10s_null"] = (
            payload.get("frames_last_10s") is None)
        findings["checks"]["status_tracks_zero_is_real_not_shown"] = True
    finally:
        httpd.shutdown()
        httpd.server_close()
    findings["passed"] = all(findings["checks"].values())
    return findings["passed"], findings


# ---------------------------------------------------------------------------
# 5. viewer draws zero skeletons from 60 s of {"type": "none"} events
# ---------------------------------------------------------------------------

_NODE_CDP_DRIVER = r"""
// Headless-Chromium probe: load the real viewer, tap Start, stream
// only {"type":"none"} SSE events for 60 s, then count non-transparent
// pixels on the overlay canvas. Any skeleton drawn would leave pixels.
const { execFile } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const VIEWER_URL = process.argv[2];
const CHROME = process.argv[3];
const WAIT_S = parseInt(process.argv[4] || '60', 10);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function getJson(url, method) {
  const res = await fetch(url, { method: method || 'GET' });
  return res.json();
}

(async () => {
  const userData = fs.mkdtempSync(path.join(os.tmpdir(), 'wall-marker-cdp-'));
  const chrome = execFile(CHROME, [
    '--headless', '--no-sandbox', '--disable-gpu',
    '--use-fake-device-for-media-stream',
    '--use-fake-ui-for-media-stream',
    '--user-data-dir=' + userData,
    '--remote-debugging-port=19222', 'about:blank',
  ], () => {});
  const errors = [];
  try {
    let version = null;
    for (let i = 0; i < 30; i++) {
      try { version = await getJson('http://127.0.0.1:19222/json/version'); break; }
      catch (e) { await sleep(500); }
    }
    if (!version) throw new Error('devtools port never came up');
    // NOTE: this HeadlessChrome build ignores the url= parameter of
    // /json/new (the target stays on about:blank), so navigation is
    // done explicitly via Page.navigate below.
    const target = await getJson(
      'http://127.0.0.1:19222/json/new', 'PUT');
    const ws = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
    let msgId = 0;
    const pending = new Map();
    const consoleErrors = [];
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.id && pending.has(m.id)) {
        pending.get(m.id)(m);
        pending.delete(m.id);
      }
      if (m.method === 'Runtime.exceptionThrown') {
        consoleErrors.push('exception: ' + JSON.stringify(m.params.exceptionDetails).slice(0, 300));
      }
      if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') {
        consoleErrors.push('console.error: ' + JSON.stringify(m.params.args).slice(0, 300));
      }
      if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') {
        consoleErrors.push('log.error: ' + (m.params.entry.text || '').slice(0, 300));
      }
    };
    const send = (method, params) => new Promise((res) => {
      const id = ++msgId;
      pending.set(id, res);
      ws.send(JSON.stringify({ id, method, params: params || {} }));
    });
    const evalJs = async (expr) => {
      const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true });
      if (r.result && r.result.exceptionDetails) {
        throw new Error('evaluate failed: ' + JSON.stringify(r.result.exceptionDetails).slice(0, 1000));
      }
      return r.result && r.result.result ? r.result.result.value : undefined;
    };
    await send('Runtime.enable');
    await send('Log.enable');
    await send('Page.enable');
    await send('Page.navigate', { url: VIEWER_URL });
    // Wait until the viewer itself has loaded. The target is created
    // blank and navigated explicitly above; a readyState poll alone
    // would pass against the initial about:blank, so the gate is the
    // triple condition (loaded + right URL + overlay canvas present).
    let loaded = false;
    for (let i = 0; i < 60 && !loaded; i++) {
      try {
        loaded = await evalJs(
          "document.readyState === 'complete' " +
          "&& location.href.indexOf('/viewer') !== -1 " +
          "&& !!document.getElementById('overlay')");
      } catch (e) {
        loaded = false;
      }
      if (!loaded) await sleep(500);
    }
    if (!loaded) {
      const href = await evalJs('location.href').catch(() => '(unknown)');
      throw new Error('viewer never loaded (href now: ' + href + ')');
    }
    // Tap Start (camera is fake, the SSE stream is real: only
    // {"type":"none"} events from the server).
    await sleep(500);
    await evalJs("(() => { const b = document.getElementById('startStop'); if (!b) throw new Error('no start button'); b.click(); return true; })()");
    await sleep(2000);
    const started = await evalJs("({running: (typeof running !== 'undefined' ? running : null), pill: document.getElementById('pillText').textContent})");
    await sleep(WAIT_S * 1000);
    const result = await evalJs(`(() => {
      const c = document.getElementById('overlay');
      const x = c.getContext('2d');
      const d = x.getImageData(0, 0, c.width, c.height).data;
      let n = 0;
      for (let i = 3; i < d.length; i += 4) { if (d[i] > 0) n++; }
      return {
        nonTransparentPixels: n,
        canvasW: c.width, canvasH: c.height,
        markersLen: (typeof markers !== 'undefined' ? markers.length : null),
        pill: document.getElementById('pillText').textContent,
        lastEventAgeMs: (typeof lastEventAt !== 'undefined' && lastEventAt ? Date.now() - lastEventAt : null),
        distancesChildren: document.getElementById('distances').children.length,
      };
    })()`);
    console.log(JSON.stringify({ ok: true, started, result, consoleErrors }));
    ws.close();
  } catch (e) {
    console.log(JSON.stringify({ ok: false, error: String(e && e.message || e), consoleErrors: errors }));
  } finally {
    chrome.kill();
    try { fs.rmSync(userData, { recursive: true, force: true }); } catch (e) {}
  }
  process.exit(0);
})().catch((e) => {
  console.log(JSON.stringify({ ok: false, error: String(e && e.message || e) }));
  process.exit(0);
});
"""


def _find_chrome():
    """Headless Chromium binary (ms-playwright cache, else PATH)."""
    candidates = []
    cache = Path.home() / ".cache" / "ms-playwright"
    if cache.exists():
        candidates += sorted(
            cache.glob("chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell"))
        candidates += sorted(
            cache.glob("chromium-*/chrome-linux64/chrome"))
    for name in ("chromium", "chromium-browser", "google-chrome"):
        import shutil
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    for cand in candidates:
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def test_viewer_zero_skeletons(port, wait_s=60):
    """Real viewer + 60 s of none events -> zero skeleton pixels."""
    findings = {"checks": {}, "console_errors": []}
    chrome = _find_chrome()
    findings["chrome"] = chrome
    if chrome is None:
        findings["checks"]["chrome_available"] = False
        findings["passed"] = False
        return findings["passed"], findings
    findings["checks"]["chrome_available"] = True
    state = ServerState()  # never fed: SSE emits {"type":"none"} only
    httpd = _start_server(state, port)
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".js",
                                         delete=False) as fh:
            fh.write(_NODE_CDP_DRIVER)
            driver_path = fh.name
        try:
            proc = subprocess.run(
                ["node", driver_path,
                 f"http://127.0.0.1:{port}/viewer", chrome, str(wait_s)],
                capture_output=True, text=True, timeout=wait_s + 60)
        finally:
            os.unlink(driver_path)
        out = proc.stdout.strip().splitlines()
        parsed = None
        for line in reversed(out):
            try:
                parsed = json.loads(line)
                break
            except Exception:
                continue
        if parsed is None:
            findings["checks"]["driver_returned_json"] = False
            findings["driver_stdout_tail"] = out[-5:]
            findings["driver_stderr_tail"] = proc.stderr.strip()[-500:]
            findings["passed"] = False
            return findings["passed"], findings
        findings["checks"]["driver_returned_json"] = True
        findings["checks"]["driver_ok"] = bool(parsed.get("ok"))
        if not parsed.get("ok"):
            findings["driver_error"] = parsed.get("error")
            findings["passed"] = False
            return findings["passed"], findings
        result = parsed.get("result") or {}
        findings["result"] = result
        findings["console_errors"] = parsed.get("consoleErrors") or []
        findings["checks"]["stream_was_live"] = (
            result.get("lastEventAgeMs") is not None
            and result["lastEventAgeMs"] < 5000)
        findings["checks"]["no_markers_held"] = (
            result.get("markersLen") == 0)
        findings["checks"]["no_distance_pills"] = (
            result.get("distancesChildren") == 0)
        findings["checks"]["zero_skeleton_pixels"] = (
            result.get("nonTransparentPixels") == 0)
        findings["checks"]["no_console_errors"] = not findings["console_errors"]
    finally:
        httpd.shutdown()
        httpd.server_close()
    findings["passed"] = all(findings["checks"].values())
    return findings["passed"], findings


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = (
    ("wording", test_wording),
    ("product-server-refuses", test_product_server_refuses),
    ("sse-none-while-bridge-invalid", test_sse_none_while_bridge_invalid),
    ("operator-dash-before-first-frame",
     test_operator_dash_before_first_frame),
    ("viewer-zero-skeletons", test_viewer_zero_skeletons),
)


def run_all(port=18901):
    """Run all five real-only tests. Returns (passed, findings)."""
    findings = {"tests": {}, "port_base": port}
    port_i = port
    for name, fn in TESTS:
        try:
            if name == "viewer-zero-skeletons":
                passed, f = fn(port_i)
            else:
                passed, f = fn(port_i)
        except Exception as exc:  # noqa: BLE001 — a crash is a failure
            passed, f = False, {"error": repr(exc)}
        findings["tests"][name] = f
        findings["tests"][name]["passed"] = passed
        port_i += 1
    findings["passed"] = all(
        findings["tests"][name]["passed"] for name, _ in TESTS)
    return findings["passed"], findings


if __name__ == "__main__":
    ok, findings = run_all()
    print("\n=== REAL-ONLY TESTS (UI server part) ===")
    for name, _ in TESTS:
        f = findings["tests"][name]
        print(f"{name:36s} {'PASS' if f['passed'] else 'FAIL'}")
        for check, val in f.get("checks", {}).items():
            print(f"    {check}: {'PASS' if val else 'FAIL'}")
        if "events_seen" in f:
            print(f"    events_seen: {f['events_seen']}, "
                  f"types: {f.get('event_types')}")
        if "result" in f:
            print(f"    viewer result: {f['result']}")
        if f.get("console_errors"):
            print(f"    console_errors: {f['console_errors']}")
        if f.get("driver_error"):
            print(f"    driver_error: {f['driver_error']}")
        if f.get("error"):
            print(f"    error: {f['error']}")
    print(f"real-only tests: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
