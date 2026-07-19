import { calculateFireRisk, PREPARED_WIND, type WindObservation } from "../../wind-data";

export const runtime = "edge";

const fallback = (reason: string) => Response.json(
  { ...PREPARED_WIND, reason },
  { headers: { "Cache-Control": "public, max-age=300" } },
);

export async function GET(request: Request) {
  const url = new URL(request.url);
  const latitude = Number(url.searchParams.get("lat"));
  const longitude = Number(url.searchParams.get("lon"));
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
    return Response.json({ error: "Valid lat and lon parameters are required." }, { status: 400 });
  }

  const apiKey = process.env.OPENWEATHER_API_KEY;
  if (!apiKey) return fallback("OPENWEATHER_API_KEY is not configured");

  try {
    const endpoint = new URL("https://api.openweathermap.org/data/2.5/weather");
    endpoint.searchParams.set("lat", String(latitude));
    endpoint.searchParams.set("lon", String(longitude));
    endpoint.searchParams.set("units", "metric");
    endpoint.searchParams.set("appid", apiKey);
    const response = await fetch(endpoint, { cache: "no-store" });
    if (!response.ok) return fallback(`OpenWeather returned ${response.status}`);

    const data = await response.json() as {
      dt?: number;
      weather?: Array<{ description?: string }>;
      main?: { temp?: number; humidity?: number };
      clouds?: { all?: number };
      rain?: { "1h"?: number };
      wind?: { speed?: number; deg?: number; gust?: number };
    };
    if (!Number.isFinite(data.wind?.speed) || !Number.isFinite(data.wind?.deg) ||
        !Number.isFinite(data.main?.temp) || !Number.isFinite(data.main?.humidity) ||
        !data.weather?.[0]?.description) {
      return fallback("OpenWeather response did not include complete current weather data");
    }

    const observation: WindObservation = {
      speedMs: data.wind!.speed!,
      directionDeg: data.wind!.deg!,
      gustMs: Number.isFinite(data.wind?.gust) ? data.wind!.gust : undefined,
      observedAt: data.dt ? new Date(data.dt * 1000).toISOString() : new Date().toISOString(),
      source: "openweather",
      temperatureC: data.main!.temp!,
      humidityPct: data.main!.humidity!,
      condition: data.weather![0].description!,
      cloudCoverPct: Number.isFinite(data.clouds?.all) ? data.clouds!.all! : 0,
      rain1hMm: Number.isFinite(data.rain?.["1h"]) ? data.rain!["1h"]! : 0,
      riskLevel: "Low",
    };
    observation.riskLevel = calculateFireRisk(observation);
    return Response.json(observation, { headers: { "Cache-Control": "public, max-age=300" } });
  } catch {
    return fallback("OpenWeather request failed");
  }
}
