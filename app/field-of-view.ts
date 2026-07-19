import type { Anchor } from "./camera-data";

export type LatLon = [number, number];

const samePoint = (a: LatLon, b: LatLon) =>
  Math.abs(a[0] - b[0]) < 1e-9 && Math.abs(a[1] - b[1]) < 1e-9;

const cross = (origin: LatLon, a: LatLon, b: LatLon) =>
  (a[1] - origin[1]) * (b[0] - origin[0]) -
  (a[0] - origin[0]) * (b[1] - origin[1]);

export function deriveFieldOfView(
  camera: LatLon,
  referencePoints: LatLon[],
  fallback: LatLon[],
): LatLon[] {
  const unique = [camera, ...referencePoints].filter(
    (point, index, points) => points.findIndex((candidate) => samePoint(candidate, point)) === index,
  );
  if (unique.length < 3) return fallback;

  const sorted = unique.slice().sort((a, b) => a[1] - b[1] || a[0] - b[0]);
  const lower: LatLon[] = [];
  for (const point of sorted) {
    while (lower.length >= 2 && cross(lower.at(-2)!, lower.at(-1)!, point) <= 0) lower.pop();
    lower.push(point);
  }
  const upper: LatLon[] = [];
  for (const point of sorted.slice().reverse()) {
    while (upper.length >= 2 && cross(upper.at(-2)!, upper.at(-1)!, point) <= 0) upper.pop();
    upper.push(point);
  }

  const hull = [...lower.slice(0, -1), ...upper.slice(0, -1)];
  return hull.length >= 3 ? hull : fallback;
}

export function projectImagePointToMap(point: [number, number], anchors: Anchor[]): LatLon | null {
  if (!anchors.length) return null;
  const nearest = anchors
    .map((anchor) => ({
      anchor,
      distanceSquared: (anchor.x - point[0]) ** 2 + (anchor.y - point[1]) ** 2,
    }))
    .sort((a, b) => a.distanceSquared - b.distanceSquared)
    .slice(0, 4);

  if (nearest[0].distanceSquared < 1e-8) {
    return [nearest[0].anchor.mapLat, nearest[0].anchor.mapLon];
  }

  let totalWeight = 0;
  let latitude = 0;
  let longitude = 0;
  for (const { anchor, distanceSquared } of nearest) {
    const weight = 1 / Math.max(distanceSquared, 0.01);
    totalWeight += weight;
    latitude += anchor.mapLat * weight;
    longitude += anchor.mapLon * weight;
  }
  return [latitude / totalWeight, longitude / totalWeight];
}
