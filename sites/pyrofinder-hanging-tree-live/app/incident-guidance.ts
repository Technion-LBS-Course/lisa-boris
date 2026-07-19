import type { Zone } from "./camera-data.ts";
import { zoneForDetection, type Detection } from "./detection-logic.ts";
import {
  cardinalDirection,
  weatherConditionsLabel,
  windDirectionLabel,
  type WindObservation,
} from "./wind-data.ts";

export function observationTime(observation: WindObservation) {
  if (!observation.observedAt) return "prepared fallback";
  return new Date(observation.observedAt).toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
  });
}

export function generalWindWatchMessage(cameraName: string, observation: WindObservation) {
  const downwindDirection = cardinalDirection(observation.directionDeg + 180);
  return `Current wind near ${cameraName} is generally toward ${downwindDirection}. Synchronized weather and fire-risk context is being considered for guidance.`;
}

export function detailedWeatherMessage(cameraName: string, observation: WindObservation) {
  const source = observation.source === "openweather" ? "OpenWeather observation" : "Prepared weather fallback";
  return `${source} for ${cameraName} at ${observationTime(observation)}: ${weatherConditionsLabel(observation)}. Wind ${windDirectionLabel(observation.directionDeg)} at ${observation.speedMs.toFixed(1)} m/s. Prototype fire-weather risk: ${observation.riskLevel.toLowerCase()}.`;
}

export function detectionAlertMessage(
  focus: Detection,
  zones: readonly Zone[],
  cameraName: string,
  downwindDirection: string,
) {
  const zone = zoneForDetection(focus, zones);
  const nextStep = focus.kind === "fire"
    ? "review the live camera feed and alert the responsible response team"
    : "review the live camera feed and inspect the downwind area";
  return `${focus.kind === "smoke" ? "Smoke" : "Fire"} detected in ${zone?.name ?? "an unmapped camera area"} in the ${cameraName} view. Downwind concern is generally toward ${downwindDirection}, ${nextStep}.`;
}
