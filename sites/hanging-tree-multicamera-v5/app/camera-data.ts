import hangingTreeMapping from "./data/hanging-tree-camera.json";
import hangingTreeResults from "./data/hanging-tree-yolo11s-results.json";
import thunderValleyResults from "./data/thunder-valley-yolo11s-results.json";

export type CameraId = "hanging-tree-1" | "thunder-valley-west";
export type Priority = "high" | "medium" | "low";
export type Detection = { kind: "smoke" | "fire"; confidence: number; box: [number, number, number, number] };
export type Anchor = { id: string; name: string; x: number; y: number; mapLat: number; mapLon: number };
export type Zone = { id: string; name: string; priority: Priority; points: [number, number][]; referencePoint?: [number, number] };
export type DetectorProvenance = {
  model: "YOLO11s";
  mode: "verified-offline";
  checkpointSha256: string;
  collectionConfidence: number;
  imgsz: number;
  generatedUtc: string;
};

export type CameraConfig = {
  id: CameraId;
  name: string;
  shortName: string;
  siteLabel: string;
  stamp: string;
  camera: [number, number];
  height: string;
  frameSize: string;
  referenceUrl: string;
  frameBase: string;
  frameCount: number;
  detections: Record<number, Detection[]>;
  anchors: Anchor[];
  zones: Zone[];
  incidentLocation: [number, number];
  incidentZone: string;
  fov: [number, number][];
  detector: DetectorProvenance;
};

type ResultFile = {
  fingerprint: {
    model: string;
    output_kind: string;
    checkpoint_sha256: string;
    collection_confidence: number;
    imgsz: number;
    generated_utc: string;
  };
  frames: Array<{
    detections: Array<{ class: string; confidence: number; bbox_norm: number[] }>;
  }>;
};

const detectorFromResults = (results: ResultFile): DetectorProvenance => {
  if (results.fingerprint.model !== "YOLO11s" || results.fingerprint.output_kind !== "verified-ultralytics-inference") {
    throw new Error("Camera results are not verified YOLO11s checkpoint outputs");
  }
  return {
    model: "YOLO11s",
    mode: "verified-offline",
    checkpointSha256: results.fingerprint.checkpoint_sha256,
    collectionConfidence: results.fingerprint.collection_confidence,
    imgsz: results.fingerprint.imgsz,
    generatedUtc: results.fingerprint.generated_utc,
  };
};

const detectionsFromResults = (results: ResultFile): Record<number, Detection[]> => Object.fromEntries(
  results.frames
    .map((frame, index) => [
      index,
      frame.detections.map(detection => ({
        kind: detection.class as Detection["kind"],
        confidence: detection.confidence,
        box: detection.bbox_norm as Detection["box"],
      })),
    ] as const)
    .filter(([, frameDetections]) => frameDetections.length > 0),
);

const thunderAnchors: Anchor[] = [
  { id: "a1", name: "1", x: 20.1, y: 81.4, mapLat: 38.84304, mapLon: -121.316998 },
  { id: "a2", name: "2", x: 37, y: 69.7, mapLat: 38.84482, mapLon: -121.316904 },
  { id: "a3", name: "3", x: 98.8, y: 55.8, mapLat: 38.857271, mapLon: -121.305292 },
  { id: "a4", name: "4", x: 66.6, y: 55.4, mapLat: 38.857259, mapLon: -121.313486 },
  { id: "a5", name: "5", x: 54.7, y: 54, mapLat: 38.86096, mapLon: -121.316912 },
  { id: "a6", name: "6", x: 38.2, y: 53.8, mapLat: 38.861799, mapLon: -121.322535 },
  { id: "a7", name: "7", x: 2.3, y: 98.8, mapLat: 38.842004, mapLon: -121.316882 },
];

const thunderZones: Zone[] = [
  { id: "z1", name: "Isolated house", priority: "high", points: [[16.5,81.4],[17.1,78.8],[21.7,78.2],[21.7,82.4]], referencePoint: [20.1, 81.4] },
  { id: "z2", name: "Lincoln Crossing", priority: "high", points: [[66.7,55.1],[97.8,55.7],[96.7,52.9],[69.6,51.6]], referencePoint: [82.7, 53.7] },
  { id: "z3", name: "Lincoln Crossing West", priority: "medium", points: [[67.4,52.2],[50.4,51.4],[34.2,53.1],[63.1,55.3]], referencePoint: [58.8, 53.1] },
];

const displayZoneName = (name: string) => ({
  "The Far Mountaions": "Distant ridge / mountains",
  "The Resdince": "Residence",
  "The North Hill": "North Hill",
}[name] ?? name);

const hangingTreeAnchors: Anchor[] = hangingTreeMapping.reference_points
  .filter(point => point.enabled)
  .map(point => ({
    id: point.point_id,
    name: point.point_name,
    x: point.image_x_norm * 100,
    y: point.image_y_norm * 100,
    mapLat: point.map_lat,
    mapLon: point.map_lon,
  }));

const hangingTreeZones: Zone[] = hangingTreeMapping.image_zones
  .filter(zone => zone.enabled)
  .map(zone => ({
    id: zone.zone_id,
    name: displayZoneName(zone.zone_name),
    priority: zone.priority_label as Priority,
    points: zone.vertices_norm.map(point => [point[0] * 100, point[1] * 100] as [number, number]),
    referencePoint: zone.zone_ref_point_norm
      ? [zone.zone_ref_point_norm[0] * 100, zone.zone_ref_point_norm[1] * 100] as [number, number]
      : undefined,
  }));

const hangingTreeDetections = detectionsFromResults(hangingTreeResults);
const thunderValleyDetections = detectionsFromResults(thunderValleyResults);

export const CAMERAS: Record<CameraId, CameraConfig> = {
  "hanging-tree-1": {
    id: "hanging-tree-1",
    name: "Hanging Tree 1",
    shortName: "CAM HT1",
    siteLabel: "ROHAN RIDGE RANCH · HANGING TREE 1",
    stamp: "HANGING TREE 1 · SIMULATED FIRE DEMO",
    camera: [35.573555, -120.667035],
    height: "5 m · Outdoor",
    frameSize: "1920 × 1080",
    referenceUrl: "/cameras/hanging-tree-1/reference.jpg",
    frameBase: "/cameras/hanging-tree-1/frames",
    frameCount: hangingTreeResults.frames.length,
    detections: hangingTreeDetections,
    anchors: hangingTreeAnchors,
    zones: hangingTreeZones,
    incidentLocation: [35.574384, -120.671347],
    incidentZone: "North Hill near the farm",
    fov: [[35.573555, -120.667035], [35.5688, -120.6945], [35.5782, -120.679]],
    detector: detectorFromResults(hangingTreeResults),
  },
  "thunder-valley-west": {
    id: "thunder-valley-west",
    name: "Thunder Valley West",
    shortName: "CAM WEST",
    siteLabel: "THUNDER VALLEY · CAMERA WEST",
    stamp: "THUNDER VALLEY WEST · LIVE DEMO",
    camera: [38.840335705966275, -121.3152325466259],
    height: "70 m · Outdoor",
    frameSize: "1920 × 1080",
    referenceUrl: "/reference.jpg",
    frameBase: "/frames",
    frameCount: thunderValleyResults.frames.length,
    detections: thunderValleyDetections,
    anchors: thunderAnchors,
    zones: thunderZones,
    incidentLocation: [38.84304, -121.316998],
    incidentZone: "Isolated house",
    fov: [[38.840335705966275, -121.3152325466259], [38.8623, -121.3275], [38.8623, -121.3028]],
    detector: detectorFromResults(thunderValleyResults),
  },
};

export const DEFAULT_CAMERA_ID: CameraId = "hanging-tree-1";

export const cameraFrameUrl = (camera: CameraConfig, index: number) =>
  `${camera.frameBase}/frame_${String(index).padStart(2, "0")}.jpg`;
