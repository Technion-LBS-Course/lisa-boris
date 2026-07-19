import React from "react";
import {
  AbsoluteFill,
  Composition,
  Easing,
  Img,
  interpolate,
  staticFile,
  useCurrentFrame,
} from "remotion";

const clamp = {
  extrapolateLeft: "clamp" as const,
  extrapolateRight: "clamp" as const,
  easing: Easing.bezier(0.16, 1, 0.3, 1),
};

const linearClamp = {
  extrapolateLeft: "clamp" as const,
  extrapolateRight: "clamp" as const,
};

const Stage = ({
  file,
  opacity,
  mask,
}: {
  file: string;
  opacity: number;
  mask: string;
}) => (
  <Img
    src={staticFile(`keyframes/${file}`)}
    style={{
      position: "absolute",
      inset: 0,
      width: "100%",
      height: "100%",
      objectFit: "cover",
      opacity,
      maskImage: mask,
      WebkitMaskImage: mask,
    }}
  />
);

const HangingTreeMockFire = () => {
  const frame = useCurrentFrame();
  const ignitionWeight = frame < 10
    ? 0
    : frame <= 14
      ? interpolate(frame, [10, 14], [0.45, 1], clamp)
      : interpolate(frame, [14, 18], [1, 0], clamp);
  const ignitionRadiusX = interpolate(frame, [10, 14], [4.5, 12], linearClamp);
  const ignitionRadiusY = interpolate(frame, [10, 14], [5, 11], linearClamp);
  const ignitionMask = `radial-gradient(ellipse ${ignitionRadiusX}% ${ignitionRadiusY}% at 47% 69%, #000 44%, transparent 100%)`;
  const growingFrontOpacity = frame < 11
    ? 0
    : interpolate(frame, [11, 21], [0.25, 1], linearClamp);
  const growthRadiusX = interpolate(frame, [11, 27], [4.5, 28], linearClamp);
  const growthRadiusY = interpolate(frame, [11, 27], [5.5, 22], linearClamp);
  const growingFrontMask = `radial-gradient(ellipse ${growthRadiusX}% ${growthRadiusY}% at 47% 70%, #000 46%, transparent 100%)`;

  return (
    <AbsoluteFill style={{ backgroundColor: "#1b211d" }}>
      <Img
        src={staticFile(`base/frame_${String(frame).padStart(2, "0")}.jpg`)}
        style={{ width: "100%", height: "100%", objectFit: "cover" }}
      />
      <Stage
        file="stage_1_v2.png"
        opacity={ignitionWeight}
        mask={ignitionMask}
      />
      <Stage
        file="stage_3_v2.png"
        opacity={growingFrontOpacity}
        mask={growingFrontMask}
      />
    </AbsoluteFill>
  );
};

export const RemotionRoot = () => (
  <Composition
    id="HangingTreeMockFire"
    component={HangingTreeMockFire}
    durationInFrames={28}
    fps={1}
    width={1920}
    height={1080}
    defaultProps={{}}
  />
);
