import assert from "node:assert/strict";
import test from "node:test";
import { deriveFieldOfView, projectImagePointToMap } from "../app/field-of-view.ts";

test("field-of-view falls back when calibration has fewer than three unique points", () => {
  const fallback = [[1, 1], [2, 2], [3, 3]];
  assert.equal(deriveFieldOfView([1, 1], [[1, 1], [2, 2]], fallback), fallback);
});

test("field-of-view returns the convex boundary and excludes interior calibration points", () => {
  const hull = deriveFieldOfView(
    [0, 0],
    [[0, 2], [2, 2], [2, 0], [1, 1], [0, 2]],
    [],
  );
  assert.equal(hull.length, 4);
  assert.deepEqual(new Set(hull.map(point => point.join(","))), new Set(["0,0", "0,2", "2,2", "2,0"]));
});

test("image-to-map projection returns the exact calibrated anchor at an exact hit", () => {
  const anchors = [
    { id: "a", name: "A", x: 25, y: 75, mapLat: 33.1, mapLon: -117.2 },
    { id: "b", name: "B", x: 75, y: 75, mapLat: 33.2, mapLon: -117.1 },
  ];
  assert.deepEqual(projectImagePointToMap([25, 75], anchors), [33.1, -117.2]);
  assert.equal(projectImagePointToMap([25, 75], []), null);
});

test("image-to-map projection interpolates between nearby anchors", () => {
  const anchors = [
    { id: "a", name: "A", x: 0, y: 0, mapLat: 0, mapLon: 0 },
    { id: "b", name: "B", x: 100, y: 0, mapLat: 0, mapLon: 10 },
  ];
  const projected = projectImagePointToMap([50, 0], anchors);
  assert.ok(projected);
  assert.ok(Math.abs(projected[0]) < 1e-12);
  assert.ok(Math.abs(projected[1] - 5) < 1e-12);
});
