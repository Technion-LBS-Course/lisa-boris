import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the PyroFinder operations prototype", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>PyroFinder — Live Operations Prototype<\/title>/i);
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

test("includes the complete prepared Hanging Tree frame sequence", () => {
  for (let index = 0; index < 28; index += 1) {
    const frame = new URL(`../public/cameras/hanging-tree-1/frames/frame_${String(index).padStart(2, "0")}.jpg`, import.meta.url);
    assert.equal(existsSync(frame), true, `missing ${frame.pathname}`);
  }
});

test("ships the hybrid basemap and leaflet-velocity integration", () => {
  const packageJson = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
  const liveMap = readFileSync(new URL("../app/LiveMap.tsx", import.meta.url), "utf8");

  assert.equal(packageJson.dependencies["leaflet-velocity"], "^2.1.4");
  assert.match(liveMap, /World_Imagery/);
  assert.match(liveMap, /World_Transportation/);
  assert.match(liveMap, /World_Boundaries_and_Places/);
  assert.match(liveMap, /velocityLayer/);
  assert.match(liveMap, /createCaliforniaWindData/);
  assert.match(liveMap, /particleMultiplier:\s*1\s*\/\s*250/);
  assert.match(liveMap, /lineWidth:\s*0\.8/);
  assert.match(liveMap, /OpenWeather current wind/);
  assert.match(liveMap, /windObservation = PREPARED_WIND/);
  assert.doesNotMatch(liveMap, /fetch\(`\/api\/wind/);
});

test("updates field of view, reveals detections progressively, and uses N=1", () => {
  const client = readFileSync(new URL("../app/LiveOpsClient.tsx", import.meta.url), "utf8");
  const fieldOfView = readFileSync(new URL("../app/field-of-view.ts", import.meta.url), "utf8");

  assert.match(client, /deriveFieldOfView/);
  assert.match(client, /projectImagePointToMap/);
  assert.match(client, /useState\(1200\)/);
  assert.match(client, /useState\(1\)/);
  assert.match(client, /setConfirmationFrames\(1\)/);
  assert.match(client, /<option value="2000">Slow<\/option>/);
  assert.match(client, /field of view updates on save/);
  assert.match(client, /revealedFireFrames/);
  assert.match(client, /index <= frame/);
  assert.match(client, /d\.kind === "fire"/);
  assert.match(fieldOfView, /const hull/);
});

test("synchronizes OpenWeather conditions, risk, map wind, and agent context", () => {
  const client = readFileSync(new URL("../app/LiveOpsClient.tsx", import.meta.url), "utf8");
  const windRoute = readFileSync(new URL("../app/api/wind/route.ts", import.meta.url), "utf8");
  const chatRoute = readFileSync(new URL("../app/api/chat/route.ts", import.meta.url), "utf8");

  assert.match(client, /windObservation=\{weatherObservation\}/);
  assert.match(client, /prototype_fire_weather_risk/);
  assert.match(client, /Refresh synchronized weather & risk/);
  assert.match(client, /generalWindWatchMessage/);
  assert.match(client, /Current wind near \$\{cameraName\} is generally toward \$\{downwindDirection\}/);
  assert.match(client, /detailedWeatherMessage/);
  assert.match(client, /Downwind concern is generally toward/);
  assert.match(windRoute, /calculateFireRisk/);
  assert.match(windRoute, /humidityPct/);
  assert.match(windRoute, /temperatureC/);
  assert.match(chatRoute, /Synchronized current weather and prototype risk/);
  assert.match(chatRoute, /Use the full weather observation silently/);
  assert.match(chatRoute, /Unless the operator explicitly asks about weather/);
});

test("includes the gradual Hanging Tree simulated-fire sequence and aligned detections", () => {
  const results = JSON.parse(readFileSync(new URL("../app/data/hanging-tree-yolo11s-results.json", import.meta.url), "utf8"));
  const remotionPackage = JSON.parse(readFileSync(new URL("../tools/mock-fire-remotion/package.json", import.meta.url), "utf8"));

  assert.equal(results.frames.slice(0, 11).every(frame => frame.detections.length === 0), true);
  assert.equal(results.frames[11].detections[0].confidence < 0.4, true);
  assert.equal(results.frames[27].detections.some(detection => detection.class === "fire" && detection.confidence >= 0.9), true);
  assert.equal(remotionPackage.dependencies.remotion, "4.0.489");
  for (let stage = 1; stage <= 3; stage += 1) {
    assert.equal(existsSync(new URL(`../tools/mock-fire-remotion/public/keyframes/stage_${stage}.jpg`, import.meta.url)), true);
  }
});
