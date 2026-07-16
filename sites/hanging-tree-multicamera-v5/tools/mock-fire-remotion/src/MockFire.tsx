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
    : frame <= 17
      ? interpolate(frame, [10, 13], [0.55, 1], clamp)
      : interpolate(frame, [17, 20], [1, 0], clamp);
  const growthWeight = frame < 17
    ? 0
    : frame <= 22
      ? interpolate(frame, [17, 20], [0, 1], clamp)
      : interpolate(frame, [22, 27], [1, 0], clamp);
  const finalWeight = frame < 22
    ? 0
    : interpolate(frame, [22, 27], [0, 0.72], clamp);

  return (
    <AbsoluteFill style={{ backgroundColor: "#1b211d" }}>
      <Img
        src={staticFile(`base/frame_${String(frame).padStart(2, "0")}.jpg`)}
        style={{ width: "100%", height: "100%", objectFit: "cover" }}
      />
      <Stage
        file="stage_1.jpg"
        opacity={ignitionWeight}
        mask="radial-gradient(ellipse 15% 13% at 47% 66%, #000 42%, transparent 100%)"
      />
      <Stage
        file="stage_2.jpg"
        opacity={growthWeight}
        mask="radial-gradient(ellipse 25% 19% at 43% 65%, #000 40%, transparent 100%)"
      />
      <Stage
        file="stage_3.jpg"
        opacity={finalWeight}
        mask="radial-gradient(ellipse 34% 23% at 39% 64%, #000 38%, transparent 100%)"
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
