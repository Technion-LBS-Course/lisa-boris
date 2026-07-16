"use client";

import { useEffect, useRef } from "react";
import "leaflet/dist/leaflet.css";
import "leaflet-velocity/dist/leaflet-velocity.css";
import { createCaliforniaWindData, PREPARED_WIND, type WindObservation } from "./wind-data";

export type MapAnchor = {
  id: string;
  name: string;
  mapLat: number;
  mapLon: number;
};

type LiveMapProps = {
  camera: [number, number];
  cameraName: string;
  cameraShortName: string;
  anchors?: MapAnchor[];
  incident?: boolean;
  incidentLocation: [number, number];
  incidentZone: string;
  fov: [number, number][];
  showWind?: boolean;
  windObservation?: WindObservation;
  interactive?: boolean;
  selectedPoint?: [number, number] | null;
  onPick?: (lat: number, lon: number) => void;
  compact?: boolean;
};

export function LiveMap({ camera, cameraName, cameraShortName, anchors = [], incident, incidentLocation, incidentZone, fov, showWind, windObservation = PREPARED_WIND, interactive, selectedPoint, onPick, compact }: LiveMapProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const onPickRef = useRef(onPick);
  useEffect(() => { onPickRef.current = onPick; }, [onPick]);

  useEffect(() => {
    let disposed = false;
    let map: import("leaflet").Map | undefined;

    void import("leaflet").then(async (L) => {
      if (disposed || !hostRef.current) return;
      map = L.map(hostRef.current, {
        center: incident ? incidentLocation : camera,
        zoom: incident ? 15 : 14,
        zoomControl: true,
        attributionControl: true,
        scrollWheelZoom: false,
      });

      const streetLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: "© OpenStreetMap contributors",
      });
      const satelliteLayer = L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        { maxZoom: 19, attribution: "Tiles © Esri, Maxar, Earthstar Geographics, and the GIS User Community" },
      ).addTo(map);
      const satelliteLabels = L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        { maxZoom: 19, attribution: "Labels © Esri" },
      ).addTo(map);
      const transportationLayer = L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}",
        { maxZoom: 19, opacity: 0.92, attribution: "Roads © Esri" },
      ).addTo(map);
      L.control.layers(
        { "Street map": streetLayer, "Satellite imagery": satelliteLayer },
        { "Roads & transportation": transportationLayer, "Places & boundaries": satelliteLabels },
        { collapsed: true, position: "topright" },
      ).addTo(map);

      const cameraIcon = L.divIcon({
        className: "pf-map-icon-shell",
        html: `<div class="pf-camera-icon">●</div><span>${cameraShortName}</span>`,
        iconSize: [112, 32], iconAnchor: [14, 16],
      });
      L.marker(camera, { icon: cameraIcon }).addTo(map).bindPopup(`${cameraName} camera`);
      L.polygon(fov, { color: "#e4522f", weight: 1.5, fillColor: "#e4522f", fillOpacity: 0.12 }).addTo(map).bindTooltip("Approximate field of view");

      anchors.forEach((anchor, index) => {
        const icon = L.divIcon({
          className: "pf-map-anchor-shell",
          html: `<div class="pf-map-anchor">${index + 1}</div>`,
          iconSize: [24, 24], iconAnchor: [12, 12],
        });
        L.marker([anchor.mapLat, anchor.mapLon], { icon }).addTo(map!).bindTooltip(anchor.name || `Anchor ${index + 1}`);
      });

      if (incident) {
        const incidentIcon = L.divIcon({
          className: "pf-map-icon-shell",
          html: '<div class="pf-incident-icon">!</div><span>APPROX. INCIDENT</span>',
          iconSize: [132, 34], iconAnchor: [15, 17],
        });
        L.marker(incidentLocation, { icon: incidentIcon }).addTo(map).bindPopup(`Approximate incident location — ${incidentZone}`);
      }

      if (showWind) {
        type VelocityLeaflet = typeof L & {
          velocityLayer?: (options: Record<string, unknown>) => import("leaflet").Layer;
        };
        const leafletGlobal = globalThis as typeof globalThis & { L?: VelocityLeaflet };
        if (!leafletGlobal.L?.velocityLayer) {
          leafletGlobal.L = L;
          await import("leaflet-velocity");
        }
        if (disposed || !map) return;
        const windPane = map.getPane("windPane") ?? map.createPane("windPane");
        windPane.style.zIndex = "430";
        windPane.style.pointerEvents = "none";
        const velocityFactory = leafletGlobal.L?.velocityLayer;
        velocityFactory?.({
          data: createCaliforniaWindData(windObservation),
          displayValues: true,
          displayOptions: {
            velocityType: windObservation.source === "openweather" ? "OpenWeather current wind" : "Prepared wind field",
            position: "bottomleft",
            emptyString: "No wind data",
            angleConvention: "meteoCW",
            showCardinal: true,
            speedUnit: "m/s",
            directionString: "Wind from",
            speedString: "Speed",
          },
          minVelocity: 0,
          maxVelocity: 10,
          velocityScale: 0.0055,
          particleAge: 58,
          particleMultiplier: 1 / 250,
          lineWidth: 0.8,
          frameRate: 18,
          colorScale: ["#dff7ff", "#c8edf6", "#addfe9", "#8bcbd9"],
          opacity: 0.52,
          paneName: "windPane",
        }).addTo(map);
      }

      if (selectedPoint) {
        const selectedIcon = L.divIcon({ className: "pf-selected-point-shell", html: '<div class="pf-selected-point">+</div>', iconSize: [28, 28], iconAnchor: [14, 14] });
        L.marker(selectedPoint, { icon: selectedIcon }).addTo(map).bindTooltip("Selected map point", { permanent: false });
      }

      if (interactive) {
        map.on("click", (event: import("leaflet").LeafletMouseEvent) => onPickRef.current?.(event.latlng.lat, event.latlng.lng));
        hostRef.current.classList.add("map-is-interactive");
      }
    });

    return () => {
      disposed = true;
      map?.remove();
    };
  }, [camera, cameraName, cameraShortName, anchors, incident, incidentLocation, incidentZone, fov, showWind, windObservation, interactive, selectedPoint]);

  return <div ref={hostRef} className={`live-leaflet-map ${compact ? "compact" : ""}`} aria-label={`${cameraName} operational map`} />;
}
