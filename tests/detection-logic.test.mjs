import assert from "node:assert/strict";
import test from "node:test";
import {
  confirmedKinds,
  detectionGeometry,
  detectionKind,
  displayedFrameNumber,
  nextAlertFocus,
  pointInPolygon,
  qualifyingDetections,
  selectAlertFocus,
  timelineMarkers,
  zoneForDetection,
} from "../app/detection-logic.ts";

const smoke = (confidence = 0.7, box = [0.5, 0.5, 0.2, 0.2]) => ({ kind: "smoke", confidence, box });
const fire = (confidence = 0.4, box = [0.5, 0.5, 0.2, 0.2]) => ({ kind: "fire", confidence, box });
const thresholds = { smoke: 0.4, fire: 0.4 };

test("filters smoke and fire independently at inclusive threshold boundaries", () => {
  assert.deepEqual(qualifyingDetections([smoke(0.4), fire(0.399)], thresholds), [smoke(0.4)]);
  assert.deepEqual(qualifyingDetections([fire(0.4)], thresholds), [fire(0.4)]);
});

test("fire outranks higher-confidence smoke without discarding either detection", () => {
  const detections = [smoke(0.99), fire(0.41), smoke(0.72)];
  assert.equal(selectAlertFocus(detections), detections[1]);
  assert.equal(selectAlertFocus([fire(0.41), fire(0.8)])?.confidence, 0.8);
  assert.equal(selectAlertFocus([]), undefined);
  assert.equal(detectionKind(detections), "both");
});

test("an active smoke alert escalates to fire and never downgrades while detections continue", () => {
  const activeSmoke = smoke(0.9);
  const detectedFire = fire(0.45);
  assert.equal(nextAlertFocus(activeSmoke, detectedFire), detectedFire);
  assert.equal(nextAlertFocus(activeSmoke, smoke(0.95)), activeSmoke);
  assert.equal(nextAlertFocus(detectedFire, fire(0.8)), detectedFire);
  assert.equal(nextAlertFocus(detectedFire, smoke(0.95)), detectedFire);
  assert.equal(nextAlertFocus(activeSmoke, undefined), null);
});

test("rejects malformed detections", () => {
  const malformed = [
    null,
    smoke(Number.NaN),
    { kind: "ember", confidence: 1, box: [0, 0, 1, 1] },
    smoke(0.5, [1, 2]),
    smoke(0.5, [1.1, 0.5, 0.2, 0.2]),
    smoke(0.5, [0.5, 0.5, 0, 0.2]),
  ];
  assert.deepEqual(qualifyingDetections(malformed, thresholds), []);
});

test("converts normalized center xywh, clips edges, and uses bottom-center anchor", () => {
  const centered = detectionGeometry(smoke(0.5, [0.5, 0.5, 0.2, 0.4]));
  assert.deepEqual(
    {
      left: Math.round(centered.left),
      top: Math.round(centered.top),
      width: Math.round(centered.width),
      height: Math.round(centered.height),
      anchor: centered.anchor.map(Math.round),
    },
    { left: 40, top: 30, width: 20, height: 40, anchor: [50, 70] },
  );

  const edged = detectionGeometry(fire(0.5, [0.05, 0.95, 0.2, 0.2]));
  assert.deepEqual(
    {
      left: edged.left,
      top: edged.top,
      width: Math.round(edged.width),
      height: Math.round(edged.height),
      anchor: edged.anchor.map(Math.round),
    },
    { left: 0, top: 85, width: 15, height: 15, anchor: [8, 100] },
  );
  assert.equal(detectionGeometry(smoke(0.5, [0.5, 0.5, 0, 0.2])), null);
});

test("matches the bottom-center anchor to polygons and falls back outside all zones", () => {
  const zone = {
    id: "z",
    name: "Calibrated",
    priority: "high",
    points: [[40, 60], [60, 60], [60, 80], [40, 80]],
  };
  assert.equal(pointInPolygon([50, 70], zone.points), true);
  assert.equal(zoneForDetection(smoke(), [zone])?.id, "z");
  assert.equal(zoneForDetection(smoke(0.7, [0.9, 0.1, 0.1, 0.1]), [zone]), null);
});

test("confirmation preserves raw-frame semantics for N=1 and consecutive N>1", () => {
  const frames = { 0: [smoke()], 1: [smoke(), fire()], 2: [smoke(), fire()] };
  assert.deepEqual(confirmedKinds(1, 1, frames, thresholds), ["fire", "smoke"]);
  assert.deepEqual(confirmedKinds(1, 2, frames, thresholds), ["smoke"]);
  assert.deepEqual(confirmedKinds(2, 2, frames, thresholds), ["fire", "smoke"]);
  assert.deepEqual(confirmedKinds(1, Number.NaN, frames, thresholds), ["fire", "smoke"]);
});

test("timeline markers use exact indexes, preserve both classes, and update thresholds", () => {
  const frames = { 2: [smoke(0.5)], 3: [smoke(0.9), fire(0.45)], 4: [fire(0.3)] };
  assert.deepEqual(timelineMarkers(frames, 3, thresholds), [
    { frame: 2, kind: "smoke", fire: false, smoke: true },
    { frame: 3, kind: "both", fire: true, smoke: true },
  ]);
  assert.deepEqual(
    timelineMarkers(frames, 3, { smoke: 0.8, fire: 0.5 }),
    [{ frame: 3, kind: "smoke", fire: false, smoke: true }],
  );
  assert.equal(displayedFrameNumber(0), 1);
  assert.equal(displayedFrameNumber(3), 4);
});
