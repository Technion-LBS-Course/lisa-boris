export type WindRecord = {
  header: {
    parameterCategory: number;
    parameterNumber: number;
    parameterNumberName: string;
    parameterUnit: string;
    forecastTime: number;
    refTime: string;
    surface1Type: number;
    surface1Value: number;
    gridDefinitionTemplate: number;
    scanMode: number;
    nx: number;
    ny: number;
    lo1: number;
    lo2: number;
    la1: number;
    la2: number;
    dx: number;
    dy: number;
  };
  data: number[];
};

export type WindObservation = {
  speedMs: number;
  directionDeg: number;
  gustMs?: number;
  observedAt?: string;
  source: "openweather" | "prepared";
  temperatureC: number;
  humidityPct: number;
  condition: string;
  cloudCoverPct: number;
  rain1hMm: number;
  riskLevel: FireRiskLevel;
};

export type FireRiskLevel = "Low" | "Moderate" | "Elevated" | "High" | "Extreme";

export const PREPARED_WIND: WindObservation = {
  speedMs: 5.15,
  directionDeg: 246,
  source: "prepared",
  temperatureC: 29,
  humidityPct: 28,
  condition: "dry and partly cloudy",
  cloudCoverPct: 38,
  rain1hMm: 0,
  riskLevel: "Elevated",
};

const CARDINAL_DIRECTIONS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"] as const;

export function cardinalDirection(directionDeg: number) {
  const normalized = ((directionDeg % 360) + 360) % 360;
  return CARDINAL_DIRECTIONS[Math.round(normalized / 22.5) % CARDINAL_DIRECTIONS.length];
}

export function windDirectionLabel(directionDeg: number) {
  return `${cardinalDirection(directionDeg)} → ${cardinalDirection(directionDeg + 180)}`;
}

export function weatherConditionsLabel(observation: WindObservation) {
  const condition = observation.condition.charAt(0).toUpperCase() + observation.condition.slice(1);
  return `${condition} · ${Math.round(observation.temperatureC)}°C · ${Math.round(observation.humidityPct)}% RH`;
}

export function calculateFireRisk(observation: Pick<WindObservation, "temperatureC" | "humidityPct" | "speedMs" | "rain1hMm">): FireRiskLevel {
  let score = 0;
  if (observation.temperatureC >= 32) score += 3;
  else if (observation.temperatureC >= 26) score += 2;
  else if (observation.temperatureC >= 20) score += 1;

  if (observation.humidityPct <= 20) score += 3;
  else if (observation.humidityPct <= 30) score += 2;
  else if (observation.humidityPct <= 40) score += 1;

  if (observation.speedMs >= 10) score += 3;
  else if (observation.speedMs >= 6) score += 2;
  else if (observation.speedMs >= 3) score += 1;

  if (observation.rain1hMm >= 1) score -= 3;
  else if (observation.rain1hMm > 0) score -= 1;

  if (score >= 8) return "Extreme";
  if (score >= 6) return "High";
  if (score >= 4) return "Elevated";
  if (score >= 2) return "Moderate";
  return "Low";
}

/**
 * Creates the GRIB2JSON shape used by leaflet-velocity. OpenWeather returns a
 * point observation as meteorological speed and "from" direction, so it is
 * converted into eastward U and northward V components across the local map.
 */
export function createCaliforniaWindData(observation: WindObservation = PREPARED_WIND): WindRecord[] {
  const nx = 25;
  const ny = 33;
  const lo1 = -125;
  const la1 = 42;
  const dx = 0.25;
  const dy = 0.25;
  const refTime = observation.observedAt ?? new Date().toISOString();
  const u: number[] = [];
  const v: number[] = [];
  const directionRadians = observation.directionDeg * Math.PI / 180;
  const baseU = -observation.speedMs * Math.sin(directionRadians);
  const baseV = -observation.speedMs * Math.cos(directionRadians);

  for (let row = 0; row < ny; row += 1) {
    for (let column = 0; column < nx; column += 1) {
      const texture = observation.source === "prepared"
        ? Math.sin((column / (nx - 1)) * Math.PI * 2) * 0.12 + Math.cos((row / (ny - 1)) * Math.PI) * 0.08
        : 0;
      u.push(Number((baseU * (1 + texture)).toFixed(3)));
      v.push(Number((baseV * (1 + texture)).toFixed(3)));
    }
  }

  const commonHeader = {
    forecastTime: 0,
    refTime,
    surface1Type: 103,
    surface1Value: 10,
    gridDefinitionTemplate: 0,
    scanMode: 0,
    nx,
    ny,
    lo1,
    lo2: lo1 + dx * (nx - 1),
    la1,
    la2: la1 - dy * (ny - 1),
    dx,
    dy,
  };

  return [
    {
      header: {
        ...commonHeader,
        parameterCategory: 2,
        parameterNumber: 2,
        parameterNumberName: "U-component of wind",
        parameterUnit: "m/s",
      },
      data: u,
    },
    {
      header: {
        ...commonHeader,
        parameterCategory: 2,
        parameterNumber: 3,
        parameterNumberName: "V-component of wind",
        parameterUnit: "m/s",
      },
      data: v,
    },
  ];
}
