export type DetectionKind = "smoke" | "fire";
export type Detection = {
  kind: DetectionKind;
  confidence: number;
  box: [number, number, number, number];
};
export type Thresholds = Record<DetectionKind, number>;
export type DetectionGeometry = {
  left: number;
  top: number;
  width: number;
  height: number;
  anchor: [number, number];
};
export type TimelineMarker = {
  frame: number;
  kind: DetectionKind | "both";
  fire: boolean;
  smoke: boolean;
};

export function validDetection(value: unknown): value is Detection {
  if (!value || typeof value !== "object") return false;
  const detection = value as Partial<Detection>;
  return (detection.kind === "smoke" || detection.kind === "fire")
    && typeof detection.confidence === "number"
    && Number.isFinite(detection.confidence)
    && detection.confidence >= 0
    && detection.confidence <= 1
    && Array.isArray(detection.box)
    && detection.box.length === 4
    && detection.box.every(coordinate =>
      typeof coordinate === "number"
      && Number.isFinite(coordinate)
      && coordinate >= 0
      && coordinate <= 1)
    && detection.box[2] > 0
    && detection.box[3] > 0;
}

export function qualifyingDetections(values: readonly unknown[], thresholds: Thresholds): Detection[] {
  return values.filter(validDetection).filter(detection => {
    const threshold = thresholds[detection.kind];
    return Number.isFinite(threshold) && threshold >= 0 && threshold <= 1
      && detection.confidence >= threshold;
  });
}

/** Fire has categorical priority; confidence only orders detections within a class. */
export function selectAlertFocus(detections: readonly Detection[]): Detection | undefined {
  return detections.slice().sort((left, right) =>
    Number(right.kind === "fire") - Number(left.kind === "fire")
      || right.confidence - left.confidence,
  )[0];
}

/** Escalate smoke to fire and keep fire priority until the incident clears. */
export function nextAlertFocus(active: Detection | null, candidate: Detection | undefined): Detection | null {
  if (!candidate) return null;
  if (active && (active.kind === "fire" || active.kind === candidate.kind)) return active;
  return candidate;
}

export function detectionKind(detections: readonly Detection[]): TimelineMarker["kind"] | undefined {
  const fire = detections.some(detection => detection.kind === "fire");
  const smoke = detections.some(detection => detection.kind === "smoke");
  return fire && smoke ? "both" : fire ? "fire" : smoke ? "smoke" : undefined;
}

/** Convert normalized center xywh into a clipped percentage rectangle. */
export function detectionGeometry(detection: Detection): DetectionGeometry | null {
  if (!validDetection(detection)) return null;
  const [centerX, centerY, width, height] = detection.box;
  const x0 = Math.max(0, Math.min(1, centerX - width / 2));
  const y0 = Math.max(0, Math.min(1, centerY - height / 2));
  const x1 = Math.max(0, Math.min(1, centerX + width / 2));
  const y1 = Math.max(0, Math.min(1, centerY + height / 2));
  if (x1 <= x0 || y1 <= y0) return null;
  return {
    left: x0 * 100,
    top: y0 * 100,
    width: (x1 - x0) * 100,
    height: (y1 - y0) * 100,
    anchor: [((x0 + x1) / 2) * 100, y1 * 100],
  };
}

export function pointInPolygon(point: [number, number], polygon: readonly [number, number][]): boolean {
  if (polygon.length < 3) return false;
  let inside = false;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const [x, y] = polygon[index];
    const [previousX, previousY] = polygon[previous];
    const crosses = (y > point[1]) !== (previousY > point[1])
      && point[0] < ((previousX - x) * (point[1] - y)) / (previousY - y) + x;
    if (crosses) inside = !inside;
  }
  return inside;
}

export function zoneForDetection<T extends { points: [number, number][] }>(
  detection: Detection,
  zones: readonly T[],
): T | null {
  const geometry = detectionGeometry(detection);
  return geometry
    ? zones.find(zone => pointInPolygon(geometry.anchor, zone.points)) ?? null
    : null;
}

export function confirmedKinds(
  frame: number,
  count: number,
  detections: Readonly<Record<number, Detection[]>>,
  thresholds: Thresholds,
): DetectionKind[] {
  const window = Number.isFinite(count) ? Math.max(1, Math.trunc(count)) : 1;
  const start = frame - window + 1;
  if (start < 0) return [];
  return (["fire", "smoke"] as const).filter(kind =>
    Array.from({ length: window }, (_, offset) => start + offset).every(index =>
      qualifyingDetections(detections[index] ?? [], thresholds)
        .some(detection => detection.kind === kind),
    ),
  );
}

export function timelineMarkers(
  detections: Readonly<Record<number, Detection[]>>,
  throughFrame: number,
  thresholds: Thresholds,
): TimelineMarker[] {
  return Object.keys(detections)
    .map(Number)
    .filter(frame => Number.isInteger(frame) && frame >= 0 && frame <= throughFrame)
    .flatMap(frame => {
      const qualifying = qualifyingDetections(detections[frame] ?? [], thresholds);
      const kind = detectionKind(qualifying);
      return kind ? [{
        frame,
        kind,
        fire: kind === "fire" || kind === "both",
        smoke: kind === "smoke" || kind === "both",
      }] : [];
    })
    .sort((left, right) => left.frame - right.frame);
}

export const displayedFrameNumber = (zeroBasedIndex: number) => zeroBasedIndex + 1;
