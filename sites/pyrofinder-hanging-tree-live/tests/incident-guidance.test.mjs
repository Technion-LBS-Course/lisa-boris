import assert from "node:assert/strict";
import test from "node:test";
import {
  detectionAlertMessage,
  detailedWeatherMessage,
  generalWindWatchMessage,
} from "../app/incident-guidance.ts";
import { PREPARED_WIND } from "../app/wind-data.ts";

const zone = {
  id: "north-hill",
  name: "North Hill",
  priority: "high",
  points: [[40, 60], [65, 60], [65, 85], [40, 85]],
};
const fire = { kind: "fire", confidence: 0.61, box: [0.52, 0.7, 0.2, 0.2] };
const smoke = { kind: "smoke", confidence: 0.7, box: [0.52, 0.7, 0.2, 0.2] };

test("fire guidance uses the detected zone, direction, and one direct next step", () => {
  const message = detectionAlertMessage(fire, [zone], "Hanging Tree 1", "ENE");
  assert.equal(
    message,
    "Fire detected in North Hill in the Hanging Tree 1 view. Downwind concern is generally toward ENE, review the live camera feed and alert the responsible response team.",
  );
  assert.doesNotMatch(message, /confirmed|approximate|confidence|\?/i);
});

test("smoke guidance recommends inspecting the downwind area without asking a question", () => {
  assert.equal(
    detectionAlertMessage(smoke, [zone], "Hanging Tree 1", "ENE"),
    "Smoke detected in North Hill in the Hanging Tree 1 view. Downwind concern is generally toward ENE, review the live camera feed and inspect the downwind area.",
  );
});

test("wind messages derive downwind direction and preserve synchronized context", () => {
  assert.equal(
    generalWindWatchMessage("Hanging Tree 1", PREPARED_WIND),
    "Current wind near Hanging Tree 1 is generally toward ENE. Synchronized weather and fire-risk context is being considered for guidance.",
  );
  const details = detailedWeatherMessage("Hanging Tree 1", PREPARED_WIND);
  assert.match(details, /^Prepared weather fallback for Hanging Tree 1 at prepared fallback:/);
  assert.match(details, /Prototype fire-weather risk: elevated\.$/);
});
