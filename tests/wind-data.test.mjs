import assert from "node:assert/strict";
import test from "node:test";
import {
  calculateFireRisk,
  cardinalDirection,
  createCaliforniaWindData,
} from "../app/wind-data.ts";

test("cardinal direction normalizes negative and wrapped bearings", () => {
  assert.equal(cardinalDirection(0), "N");
  assert.equal(cardinalDirection(66), "ENE");
  assert.equal(cardinalDirection(-90), "W");
  assert.equal(cardinalDirection(450), "E");
});

test("fire-risk score responds to heat, dryness, wind, and rain", () => {
  assert.equal(calculateFireRisk({ temperatureC: 12, humidityPct: 80, speedMs: 1, rain1hMm: 0 }), "Low");
  assert.equal(calculateFireRisk({ temperatureC: 27, humidityPct: 35, speedMs: 4, rain1hMm: 0 }), "Elevated");
  assert.equal(calculateFireRisk({ temperatureC: 35, humidityPct: 15, speedMs: 11, rain1hMm: 0 }), "Extreme");
  assert.equal(calculateFireRisk({ temperatureC: 35, humidityPct: 15, speedMs: 11, rain1hMm: 2 }), "High");
});

test("wind grid converts meteorological direction into U/V components", () => {
  const records = createCaliforniaWindData({
    speedMs: 5,
    directionDeg: 270,
    source: "openweather",
    temperatureC: 25,
    humidityPct: 40,
    condition: "clear",
    cloudCoverPct: 0,
    rain1hMm: 0,
    riskLevel: "Moderate",
  });
  assert.equal(records.length, 2);
  assert.equal(records[0].data.length, 25 * 33);
  assert.equal(records[1].data.length, 25 * 33);
  assert.equal(records[0].data.every(value => value === 5), true);
  assert.equal(records[1].data.every(value => Object.is(value, 0) || Object.is(value, -0)), true);
});
