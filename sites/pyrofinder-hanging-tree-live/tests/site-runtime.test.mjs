import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";

async function request(path = "/", init) {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${Math.random()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request(`http://localhost${path}`, init),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("built worker server-renders the PyroFinder operations app", async () => {
  const response = await request("/", { headers: { accept: "text/html" } });
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>PyroFinder.*Live Operations Prototype<\/title>/i);
  assert.match(html, /Configure the monitoring view/);
  assert.match(html, /Place camera/);
  assert.match(html, /Calibrate anchors/);
  assert.match(html, /Detection zones/);
  assert.match(html, /SYSTEM ONLINE/);
  assert.match(html, /<option value="hanging-tree-1" selected="">Hanging Tree 1<\/option>/);
  assert.match(html, /<option value="thunder-valley-west">Thunder Valley West<\/option>/);
  assert.match(html, /\/cameras\/hanging-tree-1\/reference\.jpg/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Your site is taking shape/i);
});

test("built weather route validates input and returns its prepared fallback without a key", async () => {
  const invalid = await request("/api/wind?lat=nope&lon=-117");
  assert.equal(invalid.status, 400);
  assert.deepEqual(await invalid.json(), { error: "Valid lat and lon parameters are required." });

  const savedKey = process.env.OPENWEATHER_API_KEY;
  delete process.env.OPENWEATHER_API_KEY;
  try {
    const fallback = await request("/api/wind?lat=33.37&lon=-117.16");
    assert.equal(fallback.status, 200);
    const payload = await fallback.json();
    assert.equal(payload.source, "prepared");
    assert.equal(payload.riskLevel, "Elevated");
    assert.equal(payload.reason, "OPENWEATHER_API_KEY is not configured");
  } finally {
    if (savedKey === undefined) delete process.env.OPENWEATHER_API_KEY;
    else process.env.OPENWEATHER_API_KEY = savedKey;
  }
});

test("built chat route rejects malformed requests before contacting a model", async () => {
  const response = await request("/api/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{not-json",
  });
  assert.equal(response.status, 400);
  assert.deepEqual(await response.json(), { error: "invalid_request" });
});

test("checked-in YOLO outputs match every source frame and the approved checkpoint", () => {
  const expectedCheckpoint = "6AA0C7DCD60E3572F85F02EDC05293266822F9394944479337BEBB8D178B6903";
  const resultSets = [
    {
      results: JSON.parse(readFileSync(new URL("../app/data/hanging-tree-yolo11s-results.json", import.meta.url), "utf8")),
      frameBase: new URL("../public/cameras/hanging-tree-1/frames/", import.meta.url),
      expectedFrames: 28,
    },
    {
      results: JSON.parse(readFileSync(new URL("../app/data/thunder-valley-yolo11s-results.json", import.meta.url), "utf8")),
      frameBase: new URL("../public/frames/", import.meta.url),
      expectedFrames: 26,
    },
  ];

  for (const { results, frameBase, expectedFrames } of resultSets) {
    assert.equal(results.fingerprint.model, "YOLO11s");
    assert.equal(results.fingerprint.output_kind, "verified-ultralytics-inference");
    assert.equal(results.fingerprint.checkpoint_sha256, expectedCheckpoint);
    assert.equal(results.fingerprint.checkpoint_bytes, 19151514);
    assert.deepEqual(results.fingerprint.classes, { "0": "smoke", "1": "fire" });
    assert.equal(results.fingerprint.imgsz, 640);
    assert.equal(results.fingerprint.collection_confidence, 0.05);
    assert.equal(results.fingerprint.iou, 0.5);
    assert.equal(results.frames.length, expectedFrames);
    assert.equal(results.fingerprint.n_frames, expectedFrames);
    assert.match(results.fingerprint.frame_set_sha256, /^[A-F0-9]{64}$/);
    assert.equal(new Set(results.frames.map(frame => frame.name)).size, expectedFrames);
    const frameSetHash = createHash("sha256");

    for (const [index, frame] of results.frames.entries()) {
      assert.equal(frame.name, `frame_${String(index).padStart(2, "0")}.jpg`);
      assert.equal(typeof frame.source, "string");
      assert.notEqual(frame.source.length, 0);
      assert.equal(Number.isFinite(frame.inference_ms) && frame.inference_ms >= 0, true);

      const frameBytes = readFileSync(new URL(frame.name, frameBase));
      const frameHash = createHash("sha256").update(frameBytes).digest("hex").toUpperCase();
      assert.equal(frame.sha256, frameHash, `${frame.name} provenance hash mismatch`);
      frameSetHash.update(`${frame.name}\0${frameHash}\n`);
      for (const detection of frame.detections) {
        assert.match(detection.class, /^(fire|smoke)$/);
        assert.equal(
          detection.confidence >= results.fingerprint.collection_confidence && detection.confidence <= 1,
          true,
        );
        assert.equal(detection.bbox_norm.length, 4);
        assert.equal(
          detection.bbox_norm.every(value => Number.isFinite(value) && value >= 0 && value <= 1),
          true,
        );
        assert.equal(detection.bbox_norm[2] > 0 && detection.bbox_norm[3] > 0, true);
      }
    }
    assert.equal(frameSetHash.digest("hex").toUpperCase(), results.fingerprint.frame_set_sha256);
  }
});

test("Hanging Tree fire appears, grows, and stays in the hillside corridor in authentic results", () => {
  const hangingTree = JSON.parse(
    readFileSync(new URL("../app/data/hanging-tree-yolo11s-results.json", import.meta.url), "utf8"),
  );
  const defaultThreshold = 0.4;
  const qualifyingFire = frame => frame.detections.filter(
    detection => detection.class === "fire" && detection.confidence >= defaultThreshold,
  );
  const fireArea = frame => qualifyingFire(frame).reduce(
    (sum, detection) => sum + detection.bbox_norm[2] * detection.bbox_norm[3],
    0,
  );

  const firstDefaultDetection = hangingTree.frames.findIndex(frame =>
    frame.detections.some(detection => detection.confidence >= defaultThreshold));
  const firstDefaultFire = hangingTree.frames.findIndex(frame => qualifyingFire(frame).length > 0);
  assert.equal(firstDefaultDetection, 14, "displayed frame 15 should be the first default-threshold alert");
  assert.equal(firstDefaultFire, 16, "displayed frame 17 should be the first default-threshold fire alert");
  assert.equal(
    hangingTree.frames.slice(0, 10).some(frame => frame.detections.some(detection => detection.class === "fire")),
    false,
    "pre-ignition frames must not contain fire candidates",
  );

  const matureFrames = hangingTree.frames.slice(22);
  assert.equal(matureFrames.every(frame => qualifyingFire(frame).length > 0), true);
  assert.ok(
    matureFrames.reduce((sum, frame) => sum + fireArea(frame), 0) / matureFrames.length
      > fireArea(hangingTree.frames[firstDefaultFire]) * 2,
    "mature fire should occupy more than twice the detected area of the first fire alert",
  );

  for (const [offset, frame] of hangingTree.frames.slice(firstDefaultFire).entries()) {
    const fires = qualifyingFire(frame);
    const area = fireArea(frame);
    const centerX = fires.reduce(
      (sum, detection) => sum + detection.bbox_norm[0] * detection.bbox_norm[2] * detection.bbox_norm[3],
      0,
    ) / area;
    const centerY = fires.reduce(
      (sum, detection) => sum + detection.bbox_norm[1] * detection.bbox_norm[2] * detection.bbox_norm[3],
      0,
    ) / area;
    assert.ok(centerX >= 0.45 && centerX <= 0.58, `fire left hillside corridor at frame ${firstDefaultFire + offset + 1}`);
    assert.ok(centerY >= 0.68 && centerY <= 0.76, `fire left hillside corridor at frame ${firstDefaultFire + offset + 1}`);
  }

  const frame25Fire = qualifyingFire(hangingTree.frames[24]);
  assert.ok(Math.max(...frame25Fire.map(detection => detection.confidence)) >= 0.5);
});
