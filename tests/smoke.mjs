import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

const root = process.cwd();

const requiredFiles = [
  "index.html",
  "style.css",
  "app.js",
  "Dockerfile",
  "compose.yml",
  ".github/workflows/ci.yml",
  ".github/workflows/pages.yml",
  "docs/practice-alignment.md",
  "docs/test-plan.md",
  "docs/deployment.md",
  "docs/report-outline.md",
  "assets/sample-radiology.svg"
];

requiredFiles.forEach((filePath) => {
  assert.equal(existsSync(join(root, filePath)), true, `Missing required file: ${filePath}`);
});

const indexHtml = readFileSync(join(root, "index.html"), "utf-8");
assert.match(indexHtml, /Включить камеру/, "Main camera button not found in index.html");
assert.match(indexHtml, /Калибровка 9 точек/, "Calibration control missing in index.html");
assert.match(indexHtml, /Экспорт CSV/, "CSV export control missing in index.html");
assert.match(indexHtml, /Метрики сессии/, "Metrics section missing in index.html");

const appJs = readFileSync(join(root, "app.js"), "utf-8");
assert.match(appJs, /function exportSessionAsCsv/, "CSV export logic missing in app.js");
assert.match(appJs, /function buildCalibrationModel/, "Calibration model logic missing in app.js");
assert.match(appJs, /function recordSessionPoint/, "Session recording logic missing in app.js");

const dockerfile = readFileSync(join(root, "Dockerfile"), "utf-8");
assert.match(dockerfile, /nginx:alpine/, "Dockerfile must use nginx:alpine");

console.log("Smoke test passed.");
