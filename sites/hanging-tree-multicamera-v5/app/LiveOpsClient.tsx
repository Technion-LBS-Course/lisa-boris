"use client";

import { FormEvent, MouseEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LiveMap } from "./LiveMap";
import { CAMERAS, DEFAULT_CAMERA_ID, cameraFrameUrl, type Anchor, type CameraConfig, type CameraId, type Detection, type Priority, type Zone } from "./camera-data";
import { deriveFieldOfView, projectImagePointToMap, type LatLon } from "./field-of-view";
import { cardinalDirection, PREPARED_WIND, weatherConditionsLabel, windDirectionLabel, type WindObservation } from "./wind-data";

type View = "Setup" | "Live" | "History";
type ZoneFlow = "describe" | "box" | "segmenting" | "review" | "manual" | "reference-question" | "reference-pick" | "done";
type ChatMessage = { agent?: "Watch" | "Response"; role: "assistant" | "user"; text: string };
type EventRecord = { id: string; timestamp: string; kind: "smoke" | "fire"; status: "confirmed" | "false_alarm"; confidence: number; zone: string; frame: number };

function pct(value: number) { return `${value}%`; }
function boxPolygon(points: [number, number][]): [number, number][] {
  if (points.length !== 2) return points;
  const [a, b] = points;
  return [[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]];
}
function Icon({ children }: { children: React.ReactNode }) { return <span className="icon" aria-hidden="true">{children}</span>; }

function observationTime(observation: WindObservation) {
  if (!observation.observedAt) return "prepared fallback";
  return new Date(observation.observedAt).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", timeZoneName: "short" });
}

function generalWindWatchMessage(cameraName: string, observation: WindObservation) {
  const downwindDirection = cardinalDirection(observation.directionDeg + 180);
  return `Current wind near ${cameraName} is generally toward ${downwindDirection}. Synchronized weather and fire-risk context is being considered for guidance.`;
}

function detailedWeatherMessage(cameraName: string, observation: WindObservation) {
  const source = observation.source === "openweather" ? "OpenWeather observation" : "Prepared weather fallback";
  return `${source} for ${cameraName} at ${observationTime(observation)}: ${weatherConditionsLabel(observation)}. Wind ${windDirectionLabel(observation.directionDeg)} at ${observation.speedMs.toFixed(1)} m/s. Prototype fire-weather risk: ${observation.riskLevel.toLowerCase()}.`;
}

function CameraFrame({ config, frame, detections, anchors, zones, onPick, pending, pendingMode = "box", pendingReference }: {
  config: CameraConfig;
  frame?: number;
  detections?: Detection[];
  anchors?: Anchor[];
  zones?: Zone[];
  onPick?: (x: number, y: number) => void;
  pending?: [number, number][];
  pendingMode?: "box" | "polygon";
  pendingReference?: [number, number] | null;
}) {
  const click = (event: MouseEvent<HTMLDivElement>) => {
    if (!onPick) return;
    const rect = event.currentTarget.getBoundingClientRect();
    onPick(((event.clientX - rect.left) / rect.width) * 100, ((event.clientY - rect.top) / rect.height) * 100);
  };
  const pendingShape = pendingMode === "box" ? boxPolygon(pending || []) : (pending || []);
  return (
    <div className={`camera-frame ${onPick ? "pickable" : ""}`} onClick={click}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={frame === undefined ? config.referenceUrl : cameraFrameUrl(config, frame)} alt={`${config.name} wildfire monitoring camera`} />
      {zones?.map((zone) => (
        <svg key={zone.id} className="frame-overlay" viewBox="0 0 100 100" preserveAspectRatio="none">
          <polygon points={zone.points.map(p => p.join(",")).join(" ")} className={`zone-poly ${zone.priority}`} />
          {zone.referencePoint && <circle cx={zone.referencePoint[0]} cy={zone.referencePoint[1]} r="0.9" className="zone-reference-dot" />}
        </svg>
      ))}
      {pendingShape.length > 0 && (
        <svg className="frame-overlay" viewBox="0 0 100 100" preserveAspectRatio="none">
          {pendingShape.length >= 3 ? <polygon points={pendingShape.map(p => p.join(",")).join(" ")} className="pending-line" /> : <polyline points={pendingShape.map(p => p.join(",")).join(" ")} className="pending-line" />}
        </svg>
      )}
      {pendingReference && <span className="pending-reference-dot" style={{ left: pct(pendingReference[0]), top: pct(pendingReference[1]) }}>+</span>}
      {anchors?.map(a => <span key={a.id} className="image-anchor" style={{ left: pct(a.x), top: pct(a.y) }}>{a.name}</span>)}
      {detections?.map((d, i) => {
        const [cx, cy, w, h] = d.box;
        return <div key={`${d.kind}-${i}`} className={`detection-box ${d.kind}`} style={{ left: pct((cx - w / 2) * 100), top: pct((cy - h / 2) * 100), width: pct(w * 100), height: pct(h * 100) }}><span>{d.kind.toUpperCase()} {Math.round(d.confidence * 100)}%</span></div>;
      })}
      <div className="camera-stamp">{config.stamp}</div>
    </div>
  );
}

export function LiveOpsClient() {
  const [cameraId, setCameraId] = useState<CameraId>(DEFAULT_CAMERA_ID);
  return <CameraWorkspace key={cameraId} cameraConfig={CAMERAS[cameraId]} onSelectCamera={setCameraId} />;
}

function CameraWorkspace({ cameraConfig, onSelectCamera }: { cameraConfig: CameraConfig; onSelectCamera: (cameraId: CameraId) => void }) {
  const totalFrames = cameraConfig.frameCount;
  const detections = cameraConfig.detections;
  const [view, setView] = useState<View>("Setup");
  const [setupStep, setSetupStep] = useState(1);
  const [camera, setCamera] = useState<[number, number]>(cameraConfig.camera);
  const [anchors, setAnchors] = useState(cameraConfig.anchors);
  const [zones, setZones] = useState(cameraConfig.zones);
  const [pendingImage, setPendingImage] = useState<[number, number] | null>(null);
  const [pendingMap, setPendingMap] = useState<[number, number] | null>(null);
  const [anchorName, setAnchorName] = useState("");
  const [selectedAnchorId, setSelectedAnchorId] = useState<string | null>(null);
  const [anchorEditName, setAnchorEditName] = useState("");
  const [selectedZoneId, setSelectedZoneId] = useState<string | null>(null);
  const [zoneEditName, setZoneEditName] = useState("");
  const [zoneEditPriority, setZoneEditPriority] = useState<Priority>("medium");

  const [zoneFlow, setZoneFlow] = useState<ZoneFlow>("describe");
  const [zonePoints, setZonePoints] = useState<[number, number][]>([]);
  const [segmentedPoints, setSegmentedPoints] = useState<[number, number][]>([]);
  const [zoneReference, setZoneReference] = useState<[number, number] | null>(null);
  const [zoneMapReferences, setZoneMapReferences] = useState<LatLon[]>([]);
  const [newZoneId, setNewZoneId] = useState<string | null>(null);
  const [zonePrompt, setZonePrompt] = useState("");
  const [zoneDraft, setZoneDraft] = useState<{ name: string; priority: Priority } | null>(null);
  const [zoneMessages, setZoneMessages] = useState<ChatMessage[]>([{ role: "assistant", text: "Describe what to monitor and its priority. I’ll guide you through segmentation, review, and an optional reference point." }]);

  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [smokeThreshold, setSmokeThreshold] = useState(40);
  const [fireThreshold, setFireThreshold] = useState(40);
  const [confirmationFrames, setConfirmationFrames] = useState(1);
  const [speed, setSpeed] = useState(1200);
  const [activeAlert, setActiveAlert] = useState<Detection | null>(null);
  const suppressUntilClear = useRef(false);
  const [chat, setChat] = useState<ChatMessage[]>([{ agent: "Watch", role: "assistant", text: generalWindWatchMessage(cameraConfig.name, PREPARED_WIND) }]);
  const [chatInput, setChatInput] = useState("");
  const [chatBusy, setChatBusy] = useState(false);
  const [modelMode, setModelMode] = useState<"ready" | "live" | "fallback">("ready");
  const [events, setEvents] = useState<EventRecord[]>(() => {
    if (typeof window === "undefined") return [];
    try {
      const saved = localStorage.getItem(`pyrofinder-events-${cameraConfig.id}`);
      return saved ? JSON.parse(saved) as EventRecord[] : [];
    } catch { return []; }
  });
  const [historyFilter, setHistoryFilter] = useState("All");
  const [showClip, setShowClip] = useState<number | null>(null);
  const [showWind, setShowWind] = useState(true);
  const [weatherObservation, setWeatherObservation] = useState<WindObservation>(PREPARED_WIND);
  const [weatherLoading, setWeatherLoading] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  const currentDetections = useMemo(
    () => (detections[frame] || []).filter(d => d.confidence * 100 >= (d.kind === "smoke" ? smokeThreshold : fireThreshold)),
    [detections, frame, smokeThreshold, fireThreshold],
  );
  const revealedFireFrames = useMemo(
    () => Object.keys(detections)
      .map(Number)
      .filter(index => index <= frame && (detections[index] || []).some(d => d.kind === "fire" && d.confidence * 100 >= fireThreshold)),
    [detections, frame, fireThreshold],
  );
  const currentWindLabel = windDirectionLabel(weatherObservation.directionDeg);
  const currentDownwindDirection = cardinalDirection(weatherObservation.directionDeg + 180);
  const currentConditions = weatherConditionsLabel(weatherObservation);
  const calibratedFov = useMemo(
    () => deriveFieldOfView(
      camera,
      [...anchors.map(anchor => [anchor.mapLat, anchor.mapLon] as LatLon), ...zoneMapReferences],
      cameraConfig.fov,
    ),
    [camera, anchors, zoneMapReferences, cameraConfig.fov],
  );

  const loadWeather = useCallback(async (signal?: AbortSignal) => {
    setWeatherLoading(true);
    try {
      const response = await fetch(`/api/wind?lat=${camera[0]}&lon=${camera[1]}`, { signal });
      if (!response.ok) throw new Error("weather unavailable");
      const observation = await response.json() as WindObservation;
      if (!signal?.aborted) setWeatherObservation(observation);
      return observation;
    } catch {
      if (!signal?.aborted) setWeatherObservation(PREPARED_WIND);
      return PREPARED_WIND;
    } finally {
      if (!signal?.aborted) setWeatherLoading(false);
    }
  }, [camera]);

  useEffect(() => {
    const controller = new AbortController();
    void fetch(`/api/wind?lat=${camera[0]}&lon=${camera[1]}`, { signal: controller.signal })
      .then(response => {
        if (!response.ok) throw new Error("weather unavailable");
        return response.json() as Promise<WindObservation>;
      })
      .then(observation => {
        setWeatherObservation(observation);
        setWeatherLoading(false);
        setChat(current => current.length === 1 && current[0].agent === "Watch"
          ? [{ agent: "Watch", role: "assistant", text: generalWindWatchMessage(cameraConfig.name, observation) }]
          : current);
      })
      .catch(() => {
        if (!controller.signal.aborted) setWeatherLoading(false);
      });
    return () => controller.abort();
  }, [camera, cameraConfig.name]);

  useEffect(() => {
    if (!playing || activeAlert) return;
    const timer = window.setTimeout(() => setFrame(f => f >= totalFrames - 1 ? 0 : f + 1), speed);
    return () => window.clearTimeout(timer);
  }, [playing, frame, speed, activeAlert, totalFrames]);

  useEffect(() => {
    if (activeAlert) return;
    if (!currentDetections.length) {
      suppressUntilClear.current = false;
      return;
    }
    if (suppressUntilClear.current) return;
    const start = frame - confirmationFrames + 1;
    const confirmedWindow = start >= 0 && Array.from({ length: confirmationFrames }, (_, offset) => start + offset)
      .every(index => (detections[index] || []).some(d => d.confidence * 100 >= (d.kind === "smoke" ? smokeThreshold : fireThreshold)));
    if (confirmedWindow) {
      const focus = currentDetections[0];
      queueMicrotask(() => {
        setActiveAlert(focus);
        setPlaying(false);
        setChat(c => [...c, { agent: "Response", role: "assistant", text: `${focus.kind === "smoke" ? "Smoke" : "Fire"} confirmed near ${cameraConfig.incidentZone} in the ${cameraConfig.name} view. Downwind concern is generally toward ${currentDownwindDirection}; synchronized weather and risk conditions have been considered. The map location is approximate. Confirm this incident or mark it as a false alarm?` }]);
      });
    }
  }, [frame, currentDetections, activeAlert, confirmationFrames, smokeThreshold, fireThreshold, detections, cameraConfig, currentDownwindDirection]);

  useEffect(() => { scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" }); }, [chat]);

  const resolveAlert = (status: "confirmed" | "false_alarm") => {
    if (!activeAlert) return;
    const record: EventRecord = { id: crypto.randomUUID(), timestamp: new Date().toISOString(), kind: activeAlert.kind, status, confidence: activeAlert.confidence, zone: cameraConfig.incidentZone, frame };
    const next = [...events, record];
    setEvents(next);
    try { localStorage.setItem(`pyrofinder-events-${cameraConfig.id}`, JSON.stringify(next)); } catch { /* optional */ }
    setChat(c => [...c, { role: "user", text: status === "confirmed" ? "Confirm the alert." : "Mark as a false alarm." }, { agent: "Response", role: "assistant", text: `Recorded as ${status === "confirmed" ? "confirmed" : "false alarm"}. The event is now available in History. Monitoring will resume.` }]);
    setActiveAlert(null);
    suppressUntilClear.current = true;
    setFrame(f => Math.min(f + 1, totalFrames - 1));
    setPlaying(true);
  };

  const refreshWeatherRisk = async () => {
    const observation = await loadWeather();
    setChat(c => [...c, { agent: "Watch", role: "assistant", text: detailedWeatherMessage(cameraConfig.name, observation) }]);
  };

  const deterministicReply = (value: string) => {
    const lower = value.toLowerCase();
    if (lower.includes("wind") || lower.includes("weather") || lower.includes("risk")) return detailedWeatherMessage(cameraConfig.name, weatherObservation);
    if (!activeAlert) return "No confirmed incident is active. I’m continuing to watch the camera sequence with synchronized weather and fire-risk context considered.";
    if (lower.includes("call") || lower.includes("contact")) return "For a confirmed emergency, call 911. I can draft a concise report using this camera’s approved operational context, but PyroFinder never sends or dispatches automatically.";
    if (lower.includes("where") || lower.includes("location")) return `Approximate location: ${cameraConfig.incidentZone} in the ${cameraConfig.name} view. This is a camera-projected estimate, not precise geolocation.`;
    if (lower.includes("draft") || lower.includes("message")) return `Draft: “PyroFinder ${cameraConfig.name} observed ${activeAlert.kind} near ${cameraConfig.incidentZone}. Please verify conditions and keep access clear.”`;
    return "The event is confirmed across the configured frame window. I can explain the approximate location, suggest verified contacts, or draft a notification.";
  };

  const submitChat = async (event: FormEvent) => {
    event.preventDefault();
    const value = chatInput.trim();
    if (!value || chatBusy) return;
    const userMessage: ChatMessage = { role: "user", text: value };
    const nextChat = [...chat, userMessage];
    setChat(nextChat);
    setChatInput("");
    setChatBusy(true);
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: nextChat.slice(-12).map(message => ({ role: message.role, content: message.text })),
          cameraId: cameraConfig.id,
          weather: {
            source: weatherObservation.source,
            observed_at: weatherObservation.observedAt ?? null,
            condition: weatherObservation.condition,
            temperature_c: weatherObservation.temperatureC,
            humidity_percent: weatherObservation.humidityPct,
            wind_from_degrees: weatherObservation.directionDeg,
            wind_direction: currentWindLabel,
            wind_speed_m_s: weatherObservation.speedMs,
            wind_gust_m_s: weatherObservation.gustMs ?? null,
            rain_last_hour_mm: weatherObservation.rain1hMm,
            prototype_fire_weather_risk: weatherObservation.riskLevel,
          },
          incident: activeAlert ? { class: activeAlert.kind, camera: cameraConfig.name, zone: cameraConfig.incidentZone, approximate_location: cameraConfig.incidentLocation, wind_direction: currentWindLabel, wind_speed_m_s: weatherObservation.speedMs, prototype_fire_weather_risk: weatherObservation.riskLevel, operator_status: "awaiting resolution" } : { status: "no confirmed incident" },
        }),
      });
      if (!response.ok) throw new Error("model unavailable");
      const data = await response.json() as { reply?: string };
      if (!data.reply) throw new Error("empty response");
      setModelMode("live");
      setChat(c => [...c, { agent: activeAlert ? "Response" : "Watch", role: "assistant", text: data.reply! }]);
    } catch {
      setModelMode("fallback");
      setChat(c => [...c, { agent: activeAlert ? "Response" : "Watch", role: "assistant", text: deterministicReply(value) }]);
    } finally {
      setChatBusy(false);
    }
  };

  const segmentationProposal = (points: [number, number][]) => {
    const [a, b] = points;
    const x0 = Math.min(a[0], b[0]), x1 = Math.max(a[0], b[0]);
    const y0 = Math.min(a[1], b[1]), y1 = Math.max(a[1], b[1]);
    const w = x1 - x0, h = y1 - y0;
    return [
      [x0 + w * .05, y0 + h * .18], [x0 + w * .38, y0 + h * .04],
      [x1 - w * .08, y0 + h * .12], [x1 - w * .02, y0 + h * .54],
      [x1 - w * .14, y1 - h * .06], [x0 + w * .52, y1 - h * .02],
      [x0 + w * .08, y1 - h * .14], [x0 + w * .02, y0 + h * .52],
    ] as [number, number][];
  };

  const runSegmentation = (points: [number, number][]) => {
    if (points.length !== 2) return;
    setZoneFlow("segmenting");
    setZoneMessages(m => [...m, { role: "assistant", text: "Box received. I’m segmenting the visible area inside it now…" }]);
    window.setTimeout(() => {
      setSegmentedPoints(segmentationProposal(points));
      setZoneFlow("review");
      setZoneMessages(m => [...m, { role: "assistant", text: "Here is the proposed segmented outline. Are you happy with it? If not, choose manual drawing and mark the contour yourself." }]);
    }, 550);
  };

  const submitZone = (event: FormEvent) => {
    event.preventDefault();
    const raw = zonePrompt.trim();
    if (!raw) return;
    const priority: Priority = /high|urgent|critical/i.test(raw) ? "high" : /low/i.test(raw) ? "low" : "medium";
    const cleaned = raw.replace(/\b(high|medium|low|priority|urgent|critical)\b/gi, "").replace(/[,.-]+/g, " ").replace(/\s+/g, " ").trim() || "Custom monitoring zone";
    const draft = { name: cleaned[0].toUpperCase() + cleaned.slice(1), priority };
    setZoneDraft(draft);
    setZoneFlow("box");
    setZoneMessages(m => [...m, { role: "user", text: raw }, { role: "assistant", text: `Got it — “${draft.name}” (${priority} priority). Click two opposite corners around the area and I’ll segment it.` }]);
    setZonePrompt("");
    if (zonePoints.length === 2) runSegmentation(zonePoints);
  };

  const pickZonePoint = (x: number, y: number) => {
    if (zoneFlow === "reference-pick") { setZoneReference([x, y]); return; }
    if (zoneFlow === "manual") { setZonePoints(points => [...points, [x, y]]); return; }
    if (zoneFlow !== "box") return;
    const next: [number, number][] = zonePoints.length >= 2 ? [[x, y]] : [...zonePoints, [x, y]];
    setZonePoints(next);
    if (next.length === 2 && zoneDraft) runSegmentation(next);
  };

  const saveDraftZone = (points: [number, number][]) => {
    if (!zoneDraft || points.length < 3) return;
    const id = crypto.randomUUID();
    setZones(current => [...current, { id, name: zoneDraft.name, priority: zoneDraft.priority, points }]);
    setNewZoneId(id);
    setZoneFlow("reference-question");
    setZoneMessages(m => [...m, { role: "assistant", text: `Saved “${zoneDraft.name}”. Would you like to add a reference point for this polygon?` }]);
  };

  const finishReferencePoint = () => {
    if (!newZoneId || !zoneReference) return;
    const mapReference = projectImagePointToMap(zoneReference, anchors);
    setZones(current => current.map(zone => zone.id === newZoneId ? { ...zone, referencePoint: zoneReference } : zone));
    if (mapReference) setZoneMapReferences(current => [...current, mapReference]);
    setZoneFlow("done");
    setZonePoints([]); setSegmentedPoints([]); setZoneReference(null);
    setZoneMessages(m => [...m, { role: "assistant", text: "Reference point saved. Incidents in this zone will use it for approximate map reporting, and the mapped field of view has been updated." }]);
  };

  const skipReferencePoint = () => {
    setZoneFlow("done");
    setZonePoints([]); setSegmentedPoints([]); setZoneReference(null);
    setZoneMessages(m => [...m, { role: "assistant", text: "No reference point added. This zone will use the shared image-to-map calibration." }]);
  };

  const resetZoneAssistant = () => {
    setZoneFlow("describe"); setZonePoints([]); setSegmentedPoints([]); setZoneReference(null); setNewZoneId(null); setZoneDraft(null);
    setZoneMessages([{ role: "assistant", text: "Describe the next area to monitor and its priority." }]);
  };

  const addAnchor = () => {
    if (!pendingImage || !pendingMap) return;
    setAnchors(current => [...current, { id: crypto.randomUUID(), name: anchorName.trim() || String(current.length + 1), x: pendingImage[0], y: pendingImage[1], mapLat: pendingMap[0], mapLon: pendingMap[1] }]);
    setPendingImage(null); setPendingMap(null); setAnchorName("");
  };

  const selectAnchor = (anchor: Anchor) => { setSelectedAnchorId(anchor.id); setAnchorEditName(anchor.name); };
  const saveAnchorEdit = () => {
    if (!selectedAnchorId || !anchorEditName.trim()) return;
    setAnchors(current => current.map(anchor => anchor.id === selectedAnchorId ? { ...anchor, name: anchorEditName.trim() } : anchor));
  };
  const deleteSelectedAnchor = () => {
    if (!selectedAnchorId || !window.confirm("Delete this calibration anchor?")) return;
    setAnchors(current => current.filter(anchor => anchor.id !== selectedAnchorId)); setSelectedAnchorId(null);
  };

  const selectZone = (zone: Zone) => { setSelectedZoneId(zone.id); setZoneEditName(zone.name); setZoneEditPriority(zone.priority); };
  const saveZoneEdit = () => {
    if (!selectedZoneId || !zoneEditName.trim()) return;
    setZones(current => current.map(zone => zone.id === selectedZoneId ? { ...zone, name: zoneEditName.trim(), priority: zoneEditPriority } : zone));
  };
  const deleteSelectedZone = () => {
    if (!selectedZoneId || !window.confirm("Delete this monitoring polygon?")) return;
    setZones(current => current.filter(zone => zone.id !== selectedZoneId)); setSelectedZoneId(null);
  };

  const filteredEvents = events.filter(e => historyFilter === "All" || historyFilter === "False alarms" && e.status === "false_alarm" || historyFilter.toLowerCase() === e.kind);
  const pendingZoneOverlay = zoneFlow === "review" || zoneFlow === "reference-question" || zoneFlow === "reference-pick" ? segmentedPoints : zonePoints;
  const pendingMode = zoneFlow === "manual" || zoneFlow === "review" || zoneFlow === "reference-question" || zoneFlow === "reference-pick" ? "polygon" : "box";

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => setView("Live")}><span className="brand-mark">P</span><span><strong>PyroFinder</strong><small>LIVE OPERATIONS</small></span></button>
        <nav aria-label="Primary navigation">{(["Setup", "Live", "History"] as View[]).map(item => <button key={item} className={view === item ? "active" : ""} onClick={() => setView(item)}>{item === "Setup" ? "⌁" : item === "Live" ? "◉" : "▤"} {item}</button>)}</nav>
        <div className="camera-switcher"><label htmlFor="camera-select">CAMERA</label><select id="camera-select" value={cameraConfig.id} onChange={event => onSelectCamera(event.target.value as CameraId)}><option value="hanging-tree-1">Hanging Tree 1</option><option value="thunder-valley-west">Thunder Valley West</option></select></div>
        <div className="system-state"><span className="pulse-dot" /> SYSTEM ONLINE <small>YOLO11s</small></div>
      </header>

      <main>
        {view === "Setup" && <section className="page setup-page">
          <div className="page-heading"><div><p className="eyebrow">CAMERA ONBOARDING · {cameraConfig.name.toUpperCase()}</p><h1>Configure the monitoring view</h1><p>Map the camera, calibrate image anchors, and define operational zones before going live.</p></div><button className="ghost-button" onClick={() => { setCamera(cameraConfig.camera); setAnchors(cameraConfig.anchors); setZones(cameraConfig.zones); setZoneMapReferences([]); setSelectedAnchorId(null); setSelectedZoneId(null); }}>Reset camera</button></div>
          <div className="stepper">{["Place camera", "Calibrate anchors", "Detection zones"].map((label, i) => <button key={label} className={setupStep === i + 1 ? "active" : setupStep > i + 1 ? "done" : ""} onClick={() => setSetupStep(i + 1)}><span>{setupStep > i + 1 ? "✓" : i + 1}</span><small>STEP {i + 1}</small><strong>{label}</strong></button>)}</div>

          {setupStep === 1 && <div className="setup-grid">
            <div className="panel map-card"><div className="panel-title"><span><Icon>⌖</Icon> Live camera map & field of view</span><span className="status-chip ok">CLICK TO REPOSITION</span></div><LiveMap camera={camera} cameraName={cameraConfig.name} cameraShortName={cameraConfig.shortName} anchors={anchors} incidentLocation={cameraConfig.incidentLocation} incidentZone={cameraConfig.incidentZone} fov={calibratedFov} interactive onPick={(lat, lon) => setCamera([lat, lon])} /></div>
            <div className="panel camera-card"><CameraFrame config={cameraConfig} /><div className="camera-details"><div><span>CAMERA</span><strong>{cameraConfig.name}</strong></div><div><span>LOCATION</span><strong>{camera[0].toFixed(4)}, {camera[1].toFixed(4)}</strong></div><div><span>HEIGHT</span><strong>{cameraConfig.height}</strong></div><div><span>FRAME</span><strong>{cameraConfig.frameSize}</strong></div></div><button className="primary-button" onClick={() => setSetupStep(2)}>Continue to calibration →</button></div>
          </div>}

          {setupStep === 2 && <>
            <div className="calibration-grid">
              <div className="panel"><div className="panel-title"><span><Icon>▧</Icon> Reference image</span><small>Click a recognizable point</small></div><CameraFrame config={cameraConfig} anchors={anchors} onPick={(x,y) => setPendingImage([x,y])} />{pendingImage && <p className="selection-readout">Image point selected · {pendingImage[0].toFixed(1)}%, {pendingImage[1].toFixed(1)}%</p>}</div>
              <div className="panel"><div className="panel-title"><span><Icon>⌖</Icon> Matching live map point</span><small>Click the same landmark</small></div><LiveMap compact camera={camera} cameraName={cameraConfig.name} cameraShortName={cameraConfig.shortName} anchors={anchors} incidentLocation={cameraConfig.incidentLocation} incidentZone={cameraConfig.incidentZone} fov={calibratedFov} interactive selectedPoint={pendingMap} onPick={(lat,lon) => setPendingMap([lat,lon])} />{pendingMap && <p className="selection-readout">Map point · {pendingMap[0].toFixed(5)}, {pendingMap[1].toFixed(5)}</p>}</div>
            </div>
            <div className="anchor-toolbar"><input aria-label="Anchor name" value={anchorName} onChange={e => setAnchorName(e.target.value)} placeholder="Anchor name, e.g. Water tower" /><button className="primary-button" disabled={!pendingImage || !pendingMap} onClick={addAnchor}>+ Add anchor</button><span>{anchors.length} calibrated anchors · field of view updates on save</span></div>
            <div className="anchor-list">{anchors.map((anchor, i) => <button key={anchor.id} className={selectedAnchorId === anchor.id ? "selected" : ""} onClick={() => selectAnchor(anchor)}><span>{i + 1}</span><strong>{anchor.name}</strong><small>calibrated · click to edit</small></button>)}</div>
            {selectedAnchorId && <div className="edit-panel"><div><span>EDIT ANCHOR</span><strong>Change the label or explicitly delete this anchor.</strong></div><input value={anchorEditName} onChange={e => setAnchorEditName(e.target.value)} aria-label="Edit anchor name" /><button className="primary-button" onClick={saveAnchorEdit}>Save changes</button><button className="ghost-button danger" onClick={deleteSelectedAnchor}>Delete anchor</button><button className="close-edit" onClick={() => setSelectedAnchorId(null)}>×</button></div>}
            <div className="footer-actions"><button className="ghost-button" onClick={() => setSetupStep(1)}>← Back</button><button className="primary-button" onClick={() => setSetupStep(3)}>Continue to zones →</button></div>
          </>}

          {setupStep === 3 && <>
            <div className="zones-grid">
              <div className="panel zone-workspace"><div className="panel-title"><span><Icon>◇</Icon> Monitored areas</span><small>{zoneFlow === "manual" ? "Click around the contour" : zoneFlow === "reference-pick" ? "Click the zone reference point" : "Guided segmentation workflow"}</small></div><CameraFrame config={cameraConfig} zones={zones} pending={pendingZoneOverlay} pendingMode={pendingMode} pendingReference={zoneReference} onPick={pickZonePoint} />
                <div className="zone-list">{zones.map(zone => <button key={zone.id} className={selectedZoneId === zone.id ? "selected" : ""} onClick={() => selectZone(zone)}><i className={zone.priority} /><span><strong>{zone.name}</strong><small>{zone.priority.toUpperCase()} · {zone.referencePoint ? "REFERENCE SET" : "SHARED CALIBRATION"}</small></span><b>›</b></button>)}</div>
                {selectedZoneId && <div className="edit-panel zone-edit"><div><span>EDIT POLYGON</span><strong>Selecting a polygon never deletes it.</strong></div><input value={zoneEditName} onChange={e => setZoneEditName(e.target.value)} aria-label="Edit polygon name" /><select value={zoneEditPriority} onChange={e => setZoneEditPriority(e.target.value as Priority)}><option value="high">High priority</option><option value="medium">Medium priority</option><option value="low">Low priority</option></select><button className="primary-button" onClick={saveZoneEdit}>Save changes</button><button className="ghost-button danger" onClick={deleteSelectedZone}>Delete polygon</button><button className="close-edit" onClick={() => setSelectedZoneId(null)}>×</button></div>}
              </div>
              <div className="panel assistant-panel"><div className="panel-title"><span><Icon>✦</Icon> Zone assistant</span><span className="status-chip">GUIDED WORKFLOW</span></div><div className="assistant-copy"><p className="flow-kicker">{zoneFlow.replace("-", " ").toUpperCase()}</p><h2>{zoneFlow === "review" ? "Review the segmentation." : zoneFlow === "manual" ? "Draw the contour yourself." : zoneFlow.startsWith("reference") ? "Set a reference point?" : "Describe what matters here."}</h2><p>The assistant will not save geometry until you approve it.</p></div><div className="zone-chat">{zoneMessages.map((message,i) => <div key={i} className={`bubble ${message.role}`}>{message.text}</div>)}</div>
                {zoneFlow === "segmenting" && <div className="segmentation-progress"><span /><strong>Segmenting selected area…</strong></div>}
                {zoneFlow === "review" && <div className="zone-action-card"><span>SEGMENTATION READY</span><strong>{zoneDraft?.name}</strong><p>Accept this outline or switch to manual drawing.</p><div><button className="primary-button" onClick={() => saveDraftZone(segmentedPoints)}>✓ Accept outline</button><button className="ghost-button" onClick={() => { setZoneFlow("manual"); setZonePoints([]); setSegmentedPoints([]); setZoneMessages(m => [...m,{role:"assistant",text:"Manual drawing enabled. Click at least three points around the area, then save the outline."}]); }}>Not happy · draw manually</button></div></div>}
                {zoneFlow === "manual" && <div className="zone-action-card"><span>MANUAL CONTOUR</span><strong>{zonePoints.length} points marked</strong><p>Use at least three points. Undo or reset without losing the zone description.</p><div><button className="primary-button" disabled={zonePoints.length < 3} onClick={() => saveDraftZone(zonePoints)}>Save manual outline</button><button className="ghost-button" disabled={!zonePoints.length} onClick={() => setZonePoints(points => points.slice(0,-1))}>Undo point</button><button className="ghost-button" onClick={() => setZonePoints([])}>Reset</button></div></div>}
                {zoneFlow === "reference-question" && <div className="zone-action-card"><span>OPTIONAL FINAL STEP</span><strong>Add a polygon reference point?</strong><p>This point is used for approximate map reporting when an incident falls inside the polygon.</p><div><button className="primary-button" onClick={() => { setZoneFlow("reference-pick"); setZoneMessages(m => [...m,{role:"assistant",text:"Click the reference location on the camera image, then confirm it."}]); }}>Yes · choose point</button><button className="ghost-button" onClick={skipReferencePoint}>No · use shared calibration</button></div></div>}
                {zoneFlow === "reference-pick" && <div className="zone-action-card"><span>REFERENCE POINT</span><strong>{zoneReference ? `Selected at ${zoneReference[0].toFixed(1)}%, ${zoneReference[1].toFixed(1)}%` : "Click a point on the image"}</strong><p>You may click again before saving.</p><div><button className="primary-button" disabled={!zoneReference} onClick={finishReferencePoint}>Save reference point</button><button className="ghost-button" onClick={skipReferencePoint}>Skip</button></div></div>}
                {zoneFlow === "done" && <div className="zone-action-card success"><span>ZONE COMPLETE</span><strong>{zoneDraft?.name} is ready</strong><p>The new polygon is included in monitoring.</p><button className="primary-button" onClick={resetZoneAssistant}>+ Configure another zone</button></div>}
                {(zoneFlow === "describe" || zoneFlow === "box") && <form className="chat-form" onSubmit={submitZone}><input value={zonePrompt} onChange={e => setZonePrompt(e.target.value)} placeholder="e.g. Dry brush near the house, high priority" /><button aria-label="Send">↑</button></form>}
              </div>
            </div>
            <div className="footer-actions"><button className="ghost-button" onClick={() => setSetupStep(2)}>← Back</button><button className="primary-button" onClick={() => { setView("Live"); setPlaying(true); }}>Finish setup · Go live →</button></div>
          </>}
        </section>}

        {view === "Live" && <section className="page live-page">
          <div className="live-heading"><div><p className="eyebrow">{cameraConfig.siteLabel}</p><h1>Live monitoring</h1></div><div className="weather-strip"><span>☀</span><div><small>FIRE WEATHER</small><strong>{weatherObservation.riskLevel} current risk</strong></div><div><small>WIND</small><strong>{currentWindLabel} · {weatherObservation.speedMs.toFixed(1)} m/s</strong></div><div><small>CONDITIONS</small><strong>{currentConditions}</strong></div><div className="weather-observed"><small>{weatherObservation.source === "openweather" ? "OPENWEATHER" : "PREPARED FALLBACK"}</small><strong>{observationTime(weatherObservation)}</strong></div><label className="wind-toggle"><input type="checkbox" checked={showWind} onChange={event => setShowWind(event.target.checked)} /><span>Live wind particles</span></label></div></div>
          <div className={`alert-banner ${activeAlert ? "active" : ""}`}><span className="alert-symbol">{activeAlert ? "!" : "✓"}</span><div><strong>{activeAlert ? `${activeAlert.kind.toUpperCase()} CONFIRMED · ${cameraConfig.incidentZone.toUpperCase()}` : "MONITORING · NO ACTIVE INCIDENT"}</strong><small>{activeAlert ? "Playback paused · Review approximate location and resolve in Ops chat" : "Frame sequence and weather context are being monitored"}</small></div><span className="alert-time">{activeAlert ? "ACTION REQUIRED" : "LIVE"}</span></div>
          <div className="live-grid">
            <div className="left-stage"><div className="panel camera-live"><div className="panel-title"><span><span className="live-dot" /> {cameraConfig.shortName}</span><span className="status-chip">{activeAlert ? "ALERT" : cameraConfig.id === "hanging-tree-1" ? "SIMULATED SCENARIO" : "LIVE"}</span></div><CameraFrame config={cameraConfig} frame={frame} detections={currentDetections} /><div className="playback"><button onClick={() => { setPlaying(false); setFrame(f => Math.max(0,f-1)); }} disabled={frame === 0}>‹</button><button className="play-main" onClick={() => setPlaying(p => !p)}>{playing ? "Ⅱ" : "▶"}</button><button onClick={() => { setPlaying(false); setFrame(f => Math.min(totalFrames-1,f+1)); }} disabled={frame === totalFrames-1}>›</button><div className="timeline"><i style={{ width: `${((frame + 1) / totalFrames) * 100}%` }} />{revealedFireFrames.map(index => <span key={index} style={{ left: `${(index / Math.max(1, totalFrames - 1)) * 100}%` }} />)}</div><small>{String(frame+1).padStart(2,"0")} / {totalFrames}</small></div></div><div className="panel live-map-card"><div className="panel-title"><span><Icon>⌖</Icon> Live operational map</span><small>Hybrid satellite · synchronized live wind</small></div><LiveMap camera={camera} cameraName={cameraConfig.name} cameraShortName={cameraConfig.shortName} anchors={anchors} incident={!!activeAlert} incidentLocation={cameraConfig.incidentLocation} incidentZone={cameraConfig.incidentZone} fov={calibratedFov} showWind={showWind} windObservation={weatherObservation} /></div></div>
            <aside className="ops-column"><div className="panel chat-panel"><div className="panel-title"><span><Icon>✦</Icon> Ops chat</span><span className={`status-chip model-${modelMode}`}>{modelMode === "live" ? "LIVE MODEL" : modelMode === "fallback" ? "SAFE FALLBACK" : "MODEL READY"}</span></div><div className="agent-legend"><span><b>☀</b><small>WATCH</small> Weather & risk</span><span><b>!</b><small>RESPONSE</small> Incident guidance</span></div><div className="chat-scroll" ref={scrollRef}>{chat.map((message,i) => <div key={i} className={`chat-message ${message.role}`}><small>{message.agent || "YOU"}</small><p>{message.text}</p></div>)}{chatBusy && <div className="chat-message"><small>RESPONSE</small><p className="thinking">Reading approved camera context…</p></div>}</div>{activeAlert && <div className="resolve-actions"><button className="primary-button" onClick={() => resolveAlert("confirmed")}>✓ Confirm incident</button><button className="ghost-button danger" onClick={() => resolveAlert("false_alarm")}>False alarm</button></div>}<button className="risk-button" disabled={weatherLoading} onClick={() => void refreshWeatherRisk()}>{weatherLoading ? "↻ Synchronizing weather…" : "↻ Refresh synchronized weather & risk"}</button><form className="chat-form" onSubmit={submitChat}><input value={chatInput} disabled={chatBusy} onChange={e => setChatInput(e.target.value)} placeholder={activeAlert ? "Ask about location, contacts, or a draft…" : "Ask Watch about the scene…"} /><button disabled={chatBusy} aria-label="Send">↑</button></form></div><div className="panel settings-panel"><div className="panel-title"><span><Icon>≡</Icon> Detection controls</span><button onClick={() => { setSmokeThreshold(40); setFireThreshold(40); setConfirmationFrames(1); setSpeed(1200); }}>RESET</button></div><label><span>Smoke confidence <b>{smokeThreshold}%</b></span><input type="range" min="5" max="95" step="5" value={smokeThreshold} onChange={e => setSmokeThreshold(Number(e.target.value))} /></label><label><span>Fire confidence <b>{fireThreshold}%</b></span><input type="range" min="5" max="95" step="5" value={fireThreshold} onChange={e => setFireThreshold(Number(e.target.value))} /></label><div className="settings-row"><label><span>Confirm frames (N)</span><input type="number" min="1" max="10" value={confirmationFrames} onChange={e => setConfirmationFrames(Number(e.target.value))} /></label><label><span>Speed</span><select value={speed} onChange={e => setSpeed(Number(e.target.value))}><option value="2000">Slow</option><option value="1200">Normal</option><option value="650">Fast</option></select></label></div><p>Playback uses this camera’s prepared YOLO11s outputs. Fire timeline markers appear only after a qualifying fire detection has been observed. Chat uses the same synchronized weather as the map.</p></div><div className="ops-spacer" aria-hidden="true" /></aside>
          </div>
        </section>}

        {view === "History" && <section className="page history-page"><div className="page-heading"><div><p className="eyebrow">INCIDENT REVIEW · {cameraConfig.name.toUpperCase()}</p><h1>Event history</h1><p>Resolved detections are stored on this device in this camera’s isolated history.</p></div><button className="ghost-button" onClick={() => { setEvents([]); localStorage.removeItem(`pyrofinder-events-${cameraConfig.id}`); }}>Clear history</button></div><div className="history-summary"><div><span>RESOLVED EVENTS</span><strong>{events.length}</strong><small>this camera</small></div><div><span>CONFIRMED</span><strong>{events.filter(e => e.status === "confirmed").length}</strong><small>operator verified</small></div><div><span>FALSE ALARMS</span><strong>{events.filter(e => e.status === "false_alarm").length}</strong><small>returned to monitoring</small></div><div className="mini-chart"><span>EVENT ACTIVITY</span><div>{[18,36,22,52,31,68,44,82,36,60,28,72].map((h,i) => <i key={i} style={{ height: `${h}%` }} />)}</div><small>Recent monitoring window</small></div></div><div className="history-toolbar"><div>{["All","Smoke","Fire","False alarms"].map(filter => <button key={filter} className={historyFilter === filter ? "active" : ""} onClick={() => setHistoryFilter(filter)}>{filter}</button>)}</div><span>{filteredEvents.length} events</span></div>{filteredEvents.length === 0 ? <div className="empty-history"><span>◎</span><h2>No resolved events yet</h2><p>Open Live, start playback, and resolve the detected incident to populate this timeline.</p><button className="primary-button" onClick={() => { setView("Live"); setPlaying(true); }}>Open live monitoring</button></div> : <div className="event-list">{filteredEvents.slice().reverse().map(event => <article key={event.id}><i className={event.status === "confirmed" ? "confirmed" : "false"} /><div><span>{event.kind.toUpperCase()} · {event.zone}</span><strong>{event.status === "confirmed" ? "Confirmed incident" : "False alarm"}</strong><small>{new Date(event.timestamp).toLocaleString()} · {cameraConfig.shortName} · {Math.round(event.confidence*100)}% model confidence</small></div><b className={`event-badge ${event.status}`}>{event.status === "confirmed" ? "CONFIRMED" : "FALSE ALARM"}</b><button onClick={() => setShowClip(showClip === event.frame ? null : event.frame)}>View frame</button>{showClip === event.frame && <div className="event-clip"><CameraFrame config={cameraConfig} frame={event.frame} detections={detections[event.frame]} /></div>}</article>)}</div>}<div className="scope-note"><strong>Prototype safety boundary</strong><span>Locations are approximate. PyroFinder does not predict spread, contact authorities, or dispatch automatically.</span></div></section>}
      </main>
    </div>
  );
}
