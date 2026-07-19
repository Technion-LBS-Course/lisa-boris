import { copyFile, cp, mkdir, readdir, rm } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { bundle } from "@remotion/bundler";
import { renderFrames, selectComposition } from "@remotion/renderer";

const toolRoot = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(toolRoot, "../..");
const sourceFrames = path.join(repoRoot, "public/cameras/hanging-tree-1/frames");
const cleanFireBaseFrames = path.join(toolRoot, "public/source-base");
const stagedFrames = path.join(toolRoot, "public/base");
const outputDir = path.join(toolRoot, "out");
const install = process.argv.includes("--install");

await rm(stagedFrames, { recursive: true, force: true });
await rm(outputDir, { recursive: true, force: true });
await mkdir(stagedFrames, { recursive: true });
await mkdir(outputDir, { recursive: true });
await cp(sourceFrames, stagedFrames, { recursive: true });

const cleanFireBases = (await readdir(cleanFireBaseFrames))
  .filter((file) => /^frame_(1\d|2[0-7])\.jpg$/i.test(file))
  .sort();
if (cleanFireBases.length !== 18) {
  throw new Error(`Expected 18 clean fire base frames, received ${cleanFireBases.length}`);
}
for (const file of cleanFireBases) {
  await copyFile(path.join(cleanFireBaseFrames, file), path.join(stagedFrames, file));
}

const serveUrl = await bundle({
  entryPoint: path.join(toolRoot, "src/index.tsx"),
  publicDir: path.join(toolRoot, "public"),
});
const composition = await selectComposition({
  serveUrl,
  id: "HangingTreeMockFire",
});

await renderFrames({
  serveUrl,
  composition,
  outputDir,
  imageFormat: "jpeg",
  jpegQuality: 94,
  imageSequencePattern: "frame_[frame].[ext]",
  concurrency: 4,
});

const rendered = (await readdir(outputDir))
  .filter((file) => /^frame_\d+\.jpeg$/i.test(file))
  .sort((a, b) => Number(a.match(/\d+/)?.[0]) - Number(b.match(/\d+/)?.[0]));

if (rendered.length !== 28) {
  throw new Error(`Expected 28 rendered frames, received ${rendered.length}`);
}

if (install) {
  for (let index = 10; index < rendered.length; index += 1) {
    await copyFile(
      path.join(outputDir, rendered[index]),
      path.join(sourceFrames, `frame_${String(index).padStart(2, "0")}.jpg`),
    );
  }
}

console.log(
  install
    ? "Rendered and installed Hanging Tree mock-fire frames 10-27."
    : `Rendered candidate Hanging Tree sequence to ${outputDir}. App frames were not changed.`,
);
