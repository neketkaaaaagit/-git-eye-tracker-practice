import {
  DrawingUtils,
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/+esm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm";

const LEFT_EYE = { outer: 263, inner: 362, top: 386, bottom: 374 };
const RIGHT_EYE = { outer: 33, inner: 133, top: 159, bottom: 145 };
const LEFT_IRIS = [473, 474, 475, 476, 477];
const RIGHT_IRIS = [468, 469, 470, 471, 472];

const MIN_EYE_OPENNESS = 0.018;
const BLINK_COOLDOWN_MS = 180;
const FIXATION_RADIUS = 3.2;
const FIXATION_MIN_MS = 220;
const DEFAULT_SMOOTHING = 0.2;
const RAW_HISTORY_SIZE = 12;
const AUTO_CALIBRATION_SAMPLES = 12;
const AUTO_CALIBRATION_SETTLE_MS = 550;
const AUTO_CALIBRATION_SAMPLE_INTERVAL_MS = 45;
const MIN_CALIBRATION_SPAN_X = 0.008;
const MIN_CALIBRATION_SPAN_Y = 0.006;
const AOI_KEYS = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"];

const calibrationSequence = [
  { key: "topLeft", label: "Смотрите в левый верхний угол", x: 10, y: 10 },
  { key: "topCenter", label: "Смотрите в верхнюю центральную точку", x: 50, y: 10 },
  { key: "topRight", label: "Смотрите в правый верхний угол", x: 90, y: 10 },
  { key: "middleLeft", label: "Смотрите в левую центральную точку", x: 10, y: 50 },
  { key: "center", label: "Смотрите в центр", x: 50, y: 50 },
  { key: "middleRight", label: "Смотрите в правую центральную точку", x: 90, y: 50 },
  { key: "bottomLeft", label: "Смотрите в левый нижний угол", x: 10, y: 90 },
  { key: "bottomCenter", label: "Смотрите в нижнюю центральную точку", x: 50, y: 90 },
  { key: "bottomRight", label: "Смотрите в правый нижний угол", x: 90, y: 90 },
];

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const boardPanel = document.getElementById("boardPanel");
const gazeBoard = document.getElementById("gazeBoard");
const boardImage = document.getElementById("boardImage");
const heatmapCanvas = document.getElementById("heatmapCanvas");
const heatmapCtx = heatmapCanvas.getContext("2d");
const gazeDot = document.getElementById("gazeDot");
const calibrationTarget = document.getElementById("calibrationTarget");
const boardHint = document.getElementById("boardHint");
const calibrationToolbar = document.getElementById("calibrationToolbar");
const capturePointFloatingBtn = document.getElementById("capturePointFloatingBtn");
const resetCalibrationInlineBtn = document.getElementById("resetCalibrationInlineBtn");

const statusText = document.getElementById("statusText");
const statusBadge = document.getElementById("statusBadge");
const calibrationStatus = document.getElementById("calibrationStatus");
const sessionStatus = document.getElementById("sessionStatus");
const rawValues = document.getElementById("rawValues");
const sampleCount = document.getElementById("sampleCount");
const trackedTime = document.getElementById("trackedTime");
const fixationCountEl = document.getElementById("fixationCount");
const blinkCountEl = document.getElementById("blinkCount");
const currentZoneBadge = document.getElementById("currentZoneBadge");
const fixationStateBadge = document.getElementById("fixationStateBadge");
const aoiTableBody = document.getElementById("aoiTableBody");
const toggleBoardFullscreenBtn = document.getElementById("toggleBoardFullscreenBtn");

const startCameraBtn = document.getElementById("startCameraBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const startCalibrationBtn = document.getElementById("startCalibrationBtn");
const capturePointBtn = document.getElementById("capturePointBtn");
const resetCalibrationBtn = document.getElementById("resetCalibrationBtn");
const startSessionBtn = document.getElementById("startSessionBtn");
const stopSessionBtn = document.getElementById("stopSessionBtn");
const clearSessionBtn = document.getElementById("clearSessionBtn");
const exportCsvBtn = document.getElementById("exportCsvBtn");
const exportJsonBtn = document.getElementById("exportJsonBtn");
const imageLoader = document.getElementById("imageLoader");
const heatmapOpacity = document.getElementById("heatmapOpacity");
const smoothingRange = document.getElementById("smoothingRange");

let drawingUtils;
let faceLandmarker;
let mediaStream;
let rafId;
let lastVideoTime = -1;
let lastRawPoint = null;
let rawPointHistory = [];
let smoothedPoint = { x: 50, y: 50 };
let calibrationStepIndex = -1;
let calibrationSamples = {};
let calibrationModel = null;
let isCameraRunning = false;
let isBlinkActive = false;
let blinkCount = 0;
let lastBlinkTransitionTs = 0;
let sessionActive = false;
let sessionStartTs = 0;
let lastSessionFrameTs = 0;
let sessionSamples = [];
let aoiTimes = Object.fromEntries(AOI_KEYS.map((key) => [key, 0]));
let fixationCount = 0;
let fixationCandidateStart = 0;
let fixationAnchor = null;
let isFixating = false;
let imageObjectUrl = null;
let smoothingAlpha = DEFAULT_SMOOTHING;
let heatmapStrength = 0.18;
let calibrationAutoBuffer = [];
let calibrationStepStartedTs = 0;
let lastCalibrationSampleTs = 0;

function setStatus(message, badgeMessage = message) {
  statusText.textContent = message;
  statusBadge.textContent = badgeMessage;
}

function setCalibrationStatus(message) {
  calibrationStatus.textContent = message;
}

function setSessionStatus(message) {
  sessionStatus.textContent = message;
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function lerp(from, to, alpha) {
  return from + (to - from) * alpha;
}

function getAveragePoint(landmarks, indices) {
  const sum = indices.reduce(
    (accumulator, index) => {
      accumulator.x += landmarks[index].x;
      accumulator.y += landmarks[index].y;
      return accumulator;
    },
    { x: 0, y: 0 },
  );

  return {
    x: sum.x / indices.length,
    y: sum.y / indices.length,
  };
}

function getEyeMetrics(landmarks, eye, irisIndices) {
  const outer = landmarks[eye.outer];
  const inner = landmarks[eye.inner];
  const top = landmarks[eye.top];
  const bottom = landmarks[eye.bottom];
  const iris = getAveragePoint(landmarks, irisIndices);

  const minX = Math.min(outer.x, inner.x);
  const maxX = Math.max(outer.x, inner.x);
  const minY = Math.min(top.y, bottom.y);
  const maxY = Math.max(top.y, bottom.y);
  const eyeWidth = Math.max(Math.abs(inner.x - outer.x), 0.0001);
  const eyeHeight = Math.max(Math.abs(bottom.y - top.y), 0.0001);

  return {
    xRatio: clamp01((iris.x - minX) / Math.max(maxX - minX, 0.0001)),
    yRatio: clamp01((iris.y - minY) / Math.max(maxY - minY, 0.0001)),
    openness: eyeHeight / eyeWidth,
  };
}

function updateDot(point, isCalibrated) {
  gazeDot.classList.remove("hidden");
  gazeDot.classList.toggle("calibrated", Boolean(isCalibrated));
  smoothedPoint.x = lerp(smoothedPoint.x, point.x, smoothingAlpha);
  smoothedPoint.y = lerp(smoothedPoint.y, point.y, smoothingAlpha);
  gazeDot.style.left = `${smoothedPoint.x}%`;
  gazeDot.style.top = `${smoothedPoint.y}%`;
}

function setCalibrationControlsActive(isActive) {
  calibrationToolbar.classList.toggle("hidden", !isActive);
  capturePointFloatingBtn.disabled = capturePointBtn.disabled;
}

function renderTarget(step) {
  if (!step) {
    calibrationTarget.classList.add("hidden");
    setCalibrationControlsActive(false);
    return;
  }

  calibrationTarget.classList.remove("hidden");
  calibrationTarget.style.left = `${step.x}%`;
  calibrationTarget.style.top = `${step.y}%`;
  boardHint.textContent = `${step.label}. Удерживайте взгляд на точке — сохранение выполнится автоматически.`;
  setCalibrationControlsActive(true);
}

function syncCaptureButtons() {
  capturePointFloatingBtn.disabled = capturePointBtn.disabled;
}

function resetCalibration() {
  calibrationSamples = {};
  calibrationModel = null;
  calibrationStepIndex = -1;
  calibrationAutoBuffer = [];
  calibrationStepStartedTs = 0;
  lastCalibrationSampleTs = 0;
  rawPointHistory = [];
  capturePointBtn.disabled = true;
  syncCaptureButtons();
  renderTarget(null);
  setCalibrationStatus("Не выполнена");
  boardHint.textContent =
    "Выполни калибровку, затем начни сессию и анализируй распределение внимания по зонам интереса.";
}

function beginCalibrationStep(stepIndex) {
  calibrationStepIndex = stepIndex;
  calibrationAutoBuffer = [];
  calibrationStepStartedTs = performance.now();
  lastCalibrationSampleTs = 0;
  capturePointBtn.disabled = false;
  syncCaptureButtons();
  setCalibrationStatus(`Шаг ${stepIndex + 1} из ${calibrationSequence.length} · автосбор`);
  renderTarget(calibrationSequence[stepIndex]);
}

function finalizeCalibration() {
  const model = buildCalibrationModel(calibrationSamples);
  calibrationStepIndex = -1;
  calibrationAutoBuffer = [];
  calibrationStepStartedTs = 0;
  lastCalibrationSampleTs = 0;
  capturePointBtn.disabled = true;
  syncCaptureButtons();

  if (!model) {
    renderTarget(null);
    setCalibrationStatus("Ошибка калибровки");
    boardHint.textContent =
      "Не удалось построить калибровочную модель. Попробуй снова: сядь ровнее, смотри только глазами, а не головой, и удерживай взгляд на каждой точке до автосохранения.";
    setStatus("Калибровка не удалась", "Ошибка калибровки");
    syncButtons();
    return;
  }

  calibrationModel = model;
  renderTarget(null);
  setCalibrationStatus("Готово");
  boardHint.textContent =
    "Калибровка завершена. Теперь можно запускать сессию, собирать heatmap и экспортировать данные.";
  setStatus("Калибровка завершена", "Калибровка готова");
  syncButtons();
}

function commitCalibrationPoint(point) {
  if (calibrationStepIndex < 0 || !point) {
    return;
  }

  const step = calibrationSequence[calibrationStepIndex];
  calibrationSamples[step.key] = { ...point };

  if (calibrationStepIndex >= calibrationSequence.length - 1) {
    finalizeCalibration();
    return;
  }

  beginCalibrationStep(calibrationStepIndex + 1);
}

function handleAutoCalibration(timestamp) {
  if (calibrationStepIndex < 0) {
    return;
  }

  const step = calibrationSequence[calibrationStepIndex];
  const settleProgress = timestamp - calibrationStepStartedTs;

  if (settleProgress < AUTO_CALIBRATION_SETTLE_MS) {
    const remainingSeconds = ((AUTO_CALIBRATION_SETTLE_MS - settleProgress) / 1000).toFixed(1);
    boardHint.textContent = `${step.label}. Удерживайте взгляд на точке — автосбор начнётся через ${remainingSeconds} с.`;
    return;
  }

  if (timestamp - lastCalibrationSampleTs < AUTO_CALIBRATION_SAMPLE_INTERVAL_MS) {
    boardHint.textContent = `${step.label}. Автосбор: ${calibrationAutoBuffer.length}/${AUTO_CALIBRATION_SAMPLES}`;
    return;
  }

  const stablePoint = getStableRawPoint();

  if (!stablePoint) {
    return;
  }

  calibrationAutoBuffer.push(stablePoint);
  lastCalibrationSampleTs = timestamp;
  boardHint.textContent = `${step.label}. Автосбор: ${calibrationAutoBuffer.length}/${AUTO_CALIBRATION_SAMPLES}`;

  if (calibrationAutoBuffer.length >= AUTO_CALIBRATION_SAMPLES) {
    const averagedPoint = averagePoints(calibrationAutoBuffer);
    commitCalibrationPoint(averagedPoint);
  }
}

function normalizeSegment(value, start, mid, end) {
  if (!Number.isFinite(value) || !Number.isFinite(start) || !Number.isFinite(mid) || !Number.isFinite(end)) {
    return 0.5;
  }

  const betweenStartAndMid = (value - start) * (value - mid) <= 0;

  if (betweenStartAndMid) {
    return 0.5 * clamp01((value - start) / ((mid - start) || 0.0001));
  }

  return 0.5 + 0.5 * clamp01((value - mid) / ((end - mid) || 0.0001));
}

function averageCoord(points, coordinate) {
  return points.reduce((sum, point) => sum + point[coordinate], 0) / points.length;
}

function averagePoints(points) {
  if (!points.length) {
    return null;
  }

  const averaged = points.reduce(
    (accumulator, point) => {
      accumulator.x += point.x;
      accumulator.y += point.y;
      return accumulator;
    },
    { x: 0, y: 0 },
  );

  return {
    x: averaged.x / points.length,
    y: averaged.y / points.length,
  };
}

function interpolateSegment(start, mid, end, normalizedPosition) {
  const position = clamp01(normalizedPosition);

  if (position <= 0.5) {
    return lerp(start, mid, position / 0.5);
  }

  return lerp(mid, end, (position - 0.5) / 0.5);
}

function getStableRawPoint() {
  if (!rawPointHistory.length) {
    return lastRawPoint ? { ...lastRawPoint } : null;
  }

  const averaged = rawPointHistory.reduce(
    (accumulator, point) => {
      accumulator.x += point.x;
      accumulator.y += point.y;
      return accumulator;
    },
    { x: 0, y: 0 },
  );

  return {
    x: averaged.x / rawPointHistory.length,
    y: averaged.y / rawPointHistory.length,
  };
}

function buildCalibrationModel(samples) {
  const requiredKeys = calibrationSequence.map((step) => step.key);
  const hasAllPoints = requiredKeys.every((key) => samples[key]);

  if (!hasAllPoints) {
    return null;
  }

  const leftPoints = [samples.topLeft, samples.middleLeft, samples.bottomLeft];
  const centerPoints = [samples.topCenter, samples.center, samples.bottomCenter];
  const rightPoints = [samples.topRight, samples.middleRight, samples.bottomRight];
  const topPoints = [samples.topLeft, samples.topCenter, samples.topRight];
  const middlePoints = [samples.middleLeft, samples.center, samples.middleRight];
  const bottomPoints = [samples.bottomLeft, samples.bottomCenter, samples.bottomRight];

  const model = {
    points: { ...samples },
    leftX: averageCoord(leftPoints, "x"),
    centerX: averageCoord(centerPoints, "x"),
    rightX: averageCoord(rightPoints, "x"),
    topY: averageCoord(topPoints, "y"),
    middleY: averageCoord(middlePoints, "y"),
    bottomY: averageCoord(bottomPoints, "y"),
  };

  const horizontalSpan = Math.abs(model.rightX - model.leftX);
  const verticalSpan = Math.abs(model.bottomY - model.topY);

  if (horizontalSpan < MIN_CALIBRATION_SPAN_X || verticalSpan < MIN_CALIBRATION_SPAN_Y) {
    return null;
  }

  return model;
}

function mapRawToBoard(rawPoint) {
  const safeRawX = clamp01(rawPoint.x);
  const safeRawY = clamp01(rawPoint.y);

  if (!calibrationModel) {
    return {
      x: safeRawX * 100,
      y: safeRawY * 100,
    };
  }

  const points = calibrationModel.points;
  let verticalPosition = normalizeSegment(
    safeRawY,
    calibrationModel.topY,
    calibrationModel.middleY,
    calibrationModel.bottomY,
  );

  let leftX = interpolateSegment(points.topLeft.x, points.middleLeft.x, points.bottomLeft.x, verticalPosition);
  let centerX = interpolateSegment(points.topCenter.x, points.center.x, points.bottomCenter.x, verticalPosition);
  let rightX = interpolateSegment(points.topRight.x, points.middleRight.x, points.bottomRight.x, verticalPosition);
  let horizontalPosition = normalizeSegment(safeRawX, leftX, centerX, rightX);

  const topY = interpolateSegment(points.topLeft.y, points.topCenter.y, points.topRight.y, horizontalPosition);
  const middleY = interpolateSegment(points.middleLeft.y, points.center.y, points.middleRight.y, horizontalPosition);
  const bottomY = interpolateSegment(points.bottomLeft.y, points.bottomCenter.y, points.bottomRight.y, horizontalPosition);
  verticalPosition = normalizeSegment(safeRawY, topY, middleY, bottomY);

  leftX = interpolateSegment(points.topLeft.x, points.middleLeft.x, points.bottomLeft.x, verticalPosition);
  centerX = interpolateSegment(points.topCenter.x, points.center.x, points.bottomCenter.x, verticalPosition);
  rightX = interpolateSegment(points.topRight.x, points.middleRight.x, points.bottomRight.x, verticalPosition);
  horizontalPosition = normalizeSegment(safeRawX, leftX, centerX, rightX);

  return {
    x: clamp01(horizontalPosition) * 100,
    y: clamp01(verticalPosition) * 100,
  };
}

function updateRawText(rawPoint) {
  rawValues.textContent = `x=${rawPoint.x.toFixed(3)} · y=${rawPoint.y.toFixed(3)}`;
}

function updateFullscreenButtonLabel() {
  const isFullscreen = document.fullscreenElement === boardPanel;
  boardPanel.classList.toggle("is-fullscreen", isFullscreen);
  toggleBoardFullscreenBtn.textContent = isFullscreen ? "Выйти из полного экрана" : "Во весь экран";
}

async function toggleBoardFullscreen() {
  try {
    if (document.fullscreenElement === boardPanel) {
      await document.exitFullscreen();
    } else {
      await boardPanel.requestFullscreen();
    }
  } catch (error) {
    console.error("Не удалось переключить полноэкранный режим", error);
  } finally {
    setTimeout(() => {
      resizeHeatmap();
      redrawHeatmap();
    }, 80);
  }
}

function resizeOverlay() {
  const rect = video.getBoundingClientRect();
  overlay.width = rect.width;
  overlay.height = rect.height;
}

function resizeHeatmap() {
  const rect = gazeBoard.getBoundingClientRect();
  heatmapCanvas.width = rect.width;
  heatmapCanvas.height = rect.height;
}

function drawLandmarks(landmarks) {
  if (!drawingUtils) {
    drawingUtils = new DrawingUtils(overlayCtx);
  }

  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  overlayCtx.save();
  overlayCtx.translate(overlay.width, 0);
  overlayCtx.scale(-1, 1);

  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, {
    color: "#91F4AA",
    lineWidth: 2,
  });
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, {
    color: "#91F4AA",
    lineWidth: 2,
  });
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, {
    color: "#68D4FF",
    lineWidth: 2,
  });
  drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, {
    color: "#68D4FF",
    lineWidth: 2,
  });

  overlayCtx.restore();
}

function updateBlinkState(isBlinking, timestamp) {
  if (isBlinking && !isBlinkActive && timestamp - lastBlinkTransitionTs > BLINK_COOLDOWN_MS) {
    blinkCount += 1;
    blinkCountEl.textContent = String(blinkCount);
    lastBlinkTransitionTs = timestamp;
  }

  if (isBlinkActive !== isBlinking) {
    isBlinkActive = isBlinking;
    lastBlinkTransitionTs = timestamp;
  }
}

function getZoneKey(point) {
  const colIndex = Math.min(2, Math.max(0, Math.floor(point.x / 33.3334)));
  const rowIndex = Math.min(2, Math.max(0, Math.floor(point.y / 33.3334)));
  const rowLetter = ["A", "B", "C"][rowIndex];
  const colNumber = colIndex + 1;
  return `${rowLetter}${colNumber}`;
}

function updateZoneBadge(zoneKey) {
  currentZoneBadge.textContent = `Зона: ${zoneKey}`;
}

function distanceBetween(pointA, pointB) {
  return Math.hypot(pointA.x - pointB.x, pointA.y - pointB.y);
}

function updateFixation(point, timestamp) {
  if (!fixationAnchor) {
    fixationAnchor = { ...point };
    fixationCandidateStart = timestamp;
    isFixating = false;
    fixationStateBadge.textContent = "Фиксация: нет";
    return;
  }

  const distance = distanceBetween(point, fixationAnchor);

  if (distance <= FIXATION_RADIUS) {
    fixationAnchor.x = lerp(fixationAnchor.x, point.x, 0.08);
    fixationAnchor.y = lerp(fixationAnchor.y, point.y, 0.08);

    if (!isFixating && timestamp - fixationCandidateStart >= FIXATION_MIN_MS) {
      isFixating = true;
      fixationCount += 1;
      fixationCountEl.textContent = String(fixationCount);
      fixationStateBadge.textContent = "Фиксация: да";
    }

    return;
  }

  fixationAnchor = { ...point };
  fixationCandidateStart = timestamp;
  isFixating = false;
  fixationStateBadge.textContent = "Фиксация: нет";
}

function drawHeatPoint(point) {
  const x = (point.x / 100) * heatmapCanvas.width;
  const y = (point.y / 100) * heatmapCanvas.height;
  const radius = Math.max(24, Math.min(64, heatmapCanvas.width * 0.045));

  const gradient = heatmapCtx.createRadialGradient(x, y, 6, x, y, radius);
  gradient.addColorStop(0, `rgba(255, 90, 70, ${Math.min(0.7, heatmapStrength * 2.4)})`);
  gradient.addColorStop(0.45, `rgba(255, 170, 40, ${Math.min(0.45, heatmapStrength * 1.6)})`);
  gradient.addColorStop(1, "rgba(255, 255, 0, 0)");

  heatmapCtx.fillStyle = gradient;
  heatmapCtx.beginPath();
  heatmapCtx.arc(x, y, radius, 0, Math.PI * 2);
  heatmapCtx.fill();
}

function updateAoiTable() {
  const totalTrackedSeconds = Object.values(aoiTimes).reduce((sum, value) => sum + value, 0);
  const rows = AOI_KEYS.map((zone) => {
    const seconds = aoiTimes[zone];
    const share = totalTrackedSeconds > 0 ? (seconds / totalTrackedSeconds) * 100 : 0;
    return `
      <tr>
        <td>${zone}</td>
        <td>${seconds.toFixed(1)}</td>
        <td>${share.toFixed(1)}%</td>
      </tr>
    `;
  }).join("");

  aoiTableBody.innerHTML = rows;
}

function clearHeatmap() {
  heatmapCtx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
}

function updateSessionCounters() {
  sampleCount.textContent = String(sessionSamples.length);
  const trackedSecondsValue = sessionStartTs ? Math.max(0, (lastSessionFrameTs - sessionStartTs) / 1000) : 0;
  trackedTime.textContent = `${trackedSecondsValue.toFixed(1)} c`;
}

function createTimestampLabel() {
  const now = new Date();
  const pad = (value) => String(value).padStart(2, "0");
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}_${pad(now.getHours())}-${pad(now.getMinutes())}-${pad(now.getSeconds())}`;
}

function downloadBlob(filename, blob) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function exportSessionAsCsv() {
  if (!sessionSamples.length) {
    return;
  }

  const header = ["elapsed_ms", "screen_x_pct", "screen_y_pct", "raw_x", "raw_y", "zone", "fixation"];
  const lines = sessionSamples.map((sample) => {
    return [
      sample.elapsedMs,
      sample.x.toFixed(2),
      sample.y.toFixed(2),
      sample.rawX.toFixed(4),
      sample.rawY.toFixed(4),
      sample.zone,
      sample.fixation ? 1 : 0,
    ].join(",");
  });

  const csv = [header.join(","), ...lines].join("\n");
  downloadBlob(`eye-tracking-session_${createTimestampLabel()}.csv`, new Blob([csv], { type: "text/csv;charset=utf-8" }));
}

function exportSessionAsJson() {
  if (!sessionSamples.length) {
    return;
  }

  const payload = {
    exportedAt: new Date().toISOString(),
    calibrationApplied: Boolean(calibrationModel),
    calibrationModel,
    metrics: {
      samples: sessionSamples.length,
      blinks: blinkCount,
      fixations: fixationCount,
      trackedSeconds: sessionStartTs ? Number(((lastSessionFrameTs - sessionStartTs) / 1000).toFixed(2)) : 0,
      aoiTimes,
    },
    samples: sessionSamples,
  };

  downloadBlob(
    `eye-tracking-session_${createTimestampLabel()}.json`,
    new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" }),
  );
}

function clearSessionData() {
  sessionSamples = [];
  aoiTimes = Object.fromEntries(AOI_KEYS.map((key) => [key, 0]));
  sessionStartTs = 0;
  lastSessionFrameTs = 0;
  blinkCount = 0;
  fixationCount = 0;
  fixationAnchor = null;
  fixationCandidateStart = 0;
  isFixating = false;
  blinkCountEl.textContent = "0";
  fixationCountEl.textContent = "0";
  fixationStateBadge.textContent = "Фиксация: нет";
  sampleCount.textContent = "0";
  trackedTime.textContent = "0.0 c";
  currentZoneBadge.textContent = "Зона: —";
  exportCsvBtn.disabled = true;
  exportJsonBtn.disabled = true;
  clearHeatmap();
  updateAoiTable();
}

function recordSessionPoint(boardPoint, rawPoint, timestamp, zoneKey) {
  if (!sessionActive) {
    return;
  }

  if (!sessionStartTs) {
    sessionStartTs = timestamp;
  }

  if (lastSessionFrameTs) {
    const deltaSeconds = (timestamp - lastSessionFrameTs) / 1000;
    aoiTimes[zoneKey] += Math.max(0, deltaSeconds);
  }

  lastSessionFrameTs = timestamp;
  sessionSamples.push({
    elapsedMs: Math.round(timestamp - sessionStartTs),
    x: boardPoint.x,
    y: boardPoint.y,
    rawX: rawPoint.x,
    rawY: rawPoint.y,
    zone: zoneKey,
    fixation: isFixating,
  });

  drawHeatPoint(boardPoint);
  updateAoiTable();
  updateSessionCounters();
  exportCsvBtn.disabled = false;
  exportJsonBtn.disabled = false;
}

function processLandmarks(landmarks) {
  const leftEye = getEyeMetrics(landmarks, LEFT_EYE, LEFT_IRIS);
  const rightEye = getEyeMetrics(landmarks, RIGHT_EYE, RIGHT_IRIS);
  const avgOpenness = (leftEye.openness + rightEye.openness) / 2;
  const now = performance.now();

  if (avgOpenness < MIN_EYE_OPENNESS) {
    updateBlinkState(true, now);
    setStatus("Обнаружено моргание или частично закрытые глаза", "Моргание");
    return;
  }

  updateBlinkState(false, now);

  const rawPoint = {
    x: 1 - ((leftEye.xRatio + rightEye.xRatio) / 2),
    y: (leftEye.yRatio + rightEye.yRatio) / 2,
  };

  lastRawPoint = rawPoint;
  rawPointHistory.push(rawPoint);
  if (rawPointHistory.length > RAW_HISTORY_SIZE) {
    rawPointHistory.shift();
  }
  updateRawText(rawPoint);
  handleAutoCalibration(now);

  const boardPoint = mapRawToBoard(rawPoint);
  updateDot(boardPoint, Boolean(calibrationModel));
  const zoneKey = getZoneKey(boardPoint);
  updateZoneBadge(zoneKey);
  updateFixation(boardPoint, now);
  recordSessionPoint(boardPoint, rawPoint, now, zoneKey);

  if (sessionActive) {
    setStatus(
      calibrationModel ? "Сессия активна, данные собираются" : "Сессия активна, но рекомендуется калибровка",
      "Сбор данных",
    );
    return;
  }

  if (calibrationModel) {
    setStatus("Трекинг активен", "Готово");
  } else {
    setStatus("Работает приблизительный режим без калибровки", "Без калибровки");
  }
}

async function initFaceLandmarker() {
  setStatus("Загрузка модели…", "Загрузка");
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);

  const createOptions = (delegate) => ({
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate,
    },
    runningMode: "VIDEO",
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });

  try {
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, createOptions("GPU"));
  } catch (gpuError) {
    console.warn("GPU delegate недоступен, включаю CPU", gpuError);
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, createOptions("CPU"));
  }

  setStatus("Модель готова", "Модель готова");
}

function syncButtons() {
  startCameraBtn.disabled = isCameraRunning;
  stopCameraBtn.disabled = !isCameraRunning;
  startCalibrationBtn.disabled = !isCameraRunning;
  capturePointBtn.disabled = calibrationStepIndex < 0;
  syncCaptureButtons();
  startSessionBtn.disabled = !isCameraRunning || sessionActive;
  stopSessionBtn.disabled = !sessionActive;
}

async function startCamera() {
  if (!faceLandmarker) {
    await initFaceLandmarker();
  }

  mediaStream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  });

  video.srcObject = mediaStream;

  await new Promise((resolve) => {
    video.onloadedmetadata = () => resolve();
  });

  await video.play();
  resizeOverlay();
  resizeHeatmap();

  isCameraRunning = true;
  lastVideoTime = -1;
  syncButtons();
  setStatus("Камера включена", "Камера активна");
  renderLoop();
}

function stopCamera() {
  if (sessionActive) {
    stopSession();
  }

  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  video.srcObject = null;
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  rawPointHistory = [];
  isCameraRunning = false;
  syncButtons();
  setStatus("Камера выключена", "Камера выключена");
}

function renderLoop() {
  if (!isCameraRunning || !faceLandmarker) {
    return;
  }

  if (video.currentTime !== lastVideoTime) {
    const results = faceLandmarker.detectForVideo(video, performance.now());
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

    if (results.faceLandmarks?.length) {
      const landmarks = results.faceLandmarks[0];
      drawLandmarks(landmarks);
      processLandmarks(landmarks);
    } else {
      fixationStateBadge.textContent = "Фиксация: нет";
      setStatus("Лицо не найдено", "Лицо не найдено");
    }

    lastVideoTime = video.currentTime;
  }

  rafId = requestAnimationFrame(renderLoop);
}

function startCalibration() {
  if (!lastRawPoint) {
    alert("Сначала включи камеру и дождись, пока лицо будет распознано.");
    return;
  }

  calibrationSamples = {};
  calibrationModel = null;
  calibrationStepIndex = 0;
  capturePointBtn.disabled = false;
  syncCaptureButtons();
  setCalibrationStatus(`Шаг 1 из ${calibrationSequence.length}`);
  renderTarget(calibrationSequence[calibrationStepIndex]);
  boardHint.textContent = "Перед сохранением каждой точки задержи взгляд на цели примерно на 0.5–1 секунды, затем нажми «Сохранить точку».";
  setStatus("Калибровка запущена", "Калибровка");
  gazeBoard.scrollIntoView({ behavior: "smooth", block: "center" });
}

function saveCalibrationPoint() {
  const stableRawPoint = getStableRawPoint();

  if (calibrationStepIndex < 0 || !stableRawPoint) {
    return;
  }

  const step = calibrationSequence[calibrationStepIndex];
  calibrationSamples[step.key] = { ...stableRawPoint };
  calibrationStepIndex += 1;

  if (calibrationStepIndex >= calibrationSequence.length) {
    const model = buildCalibrationModel(calibrationSamples);
    calibrationStepIndex = -1;
    capturePointBtn.disabled = true;
    syncCaptureButtons();

    if (!model) {
      renderTarget(null);
      setCalibrationStatus("Ошибка калибровки");
      boardHint.textContent =
        "Не удалось построить калибровочную модель: разница между точками получилась слишком маленькой. Попробуй смотреть поочерёдно точно в углы и центр, задерживаясь 0.5–1 секунды перед сохранением.";
      setStatus("Калибровка не удалась: маленький разброс точек", "Ошибка калибровки");
      syncButtons();
      return;
    }

    calibrationModel = model;
    renderTarget(null);
    setCalibrationStatus("Готово");
    boardHint.textContent =
      "Калибровка завершена. Теперь можно запускать сессию, собирать heatmap и экспортировать данные.";
    setStatus("Калибровка завершена", "Калибровка готова");
    syncButtons();
    return;
  }

  capturePointBtn.disabled = false;
  syncCaptureButtons();
  setCalibrationStatus(`Шаг ${calibrationStepIndex + 1} из ${calibrationSequence.length}`);
  renderTarget(calibrationSequence[calibrationStepIndex]);
}

function startSession() {
  if (!isCameraRunning) {
    alert("Сначала включи камеру.");
    return;
  }

  if (!calibrationModel) {
    const continueWithoutCalibration = window.confirm(
      "Калибровка не выполнена. Продолжить в приблизительном режиме?",
    );

    if (!continueWithoutCalibration) {
      return;
    }
  }

  sessionActive = true;
  sessionStartTs = 0;
  lastSessionFrameTs = 0;
  setSessionStatus("Идёт сбор данных");
  boardHint.textContent =
    "Сессия активна. Смотри на области изображения, затем выгрузи CSV/JSON для анализа и отчёта.";
  syncButtons();
}

function stopSession() {
  sessionActive = false;
  setSessionStatus(sessionSamples.length ? "Остановлена, данные сохранены в памяти страницы" : "Не запущена");
  syncButtons();
}

function handleImageUpload(event) {
  const [file] = event.target.files || [];

  if (!file) {
    return;
  }

  if (imageObjectUrl) {
    URL.revokeObjectURL(imageObjectUrl);
  }

  imageObjectUrl = URL.createObjectURL(file);
  boardImage.src = imageObjectUrl;
  boardHint.textContent =
    "Изображение обновлено. Выполни калибровку и запусти сессию для анализа зон внимания на новом материале.";
}

function handleHeatmapStrengthChange() {
  heatmapStrength = Number(heatmapOpacity.value) / 100;
}

function handleSmoothingChange() {
  smoothingAlpha = Number(smoothingRange.value) / 100;
}

startCameraBtn.addEventListener("click", async () => {
  try {
    await startCamera();
  } catch (error) {
    console.error(error);
    setStatus("Не удалось получить доступ к камере", "Ошибка камеры");
    alert("Не удалось запустить камеру. Проверь доступ к камере и открой приложение через localhost или HTTPS.");
  }
});

stopCameraBtn.addEventListener("click", stopCamera);
startCalibrationBtn.addEventListener("click", startCalibration);
capturePointBtn.addEventListener("click", saveCalibrationPoint);
resetCalibrationBtn.addEventListener("click", resetCalibration);
resetCalibrationInlineBtn.addEventListener("click", resetCalibration);
capturePointFloatingBtn.addEventListener("click", saveCalibrationPoint);
startSessionBtn.addEventListener("click", startSession);
stopSessionBtn.addEventListener("click", stopSession);
clearSessionBtn.addEventListener("click", clearSessionData);
exportCsvBtn.addEventListener("click", exportSessionAsCsv);
exportJsonBtn.addEventListener("click", exportSessionAsJson);
toggleBoardFullscreenBtn.addEventListener("click", toggleBoardFullscreen);
imageLoader.addEventListener("change", handleImageUpload);
heatmapOpacity.addEventListener("input", handleHeatmapStrengthChange);
smoothingRange.addEventListener("input", handleSmoothingChange);

window.addEventListener("resize", () => {
  resizeOverlay();
  resizeHeatmap();
});

document.addEventListener("keydown", (event) => {
  if (calibrationStepIndex < 0 || capturePointBtn.disabled) {
    return;
  }

  if (event.code === "Space" || event.code === "Enter" || event.code === "NumpadEnter") {
    event.preventDefault();
    saveCalibrationPoint();
  }
});
window.addEventListener("beforeunload", () => {
  if (imageObjectUrl) {
    URL.revokeObjectURL(imageObjectUrl);
  }
  stopCamera();
});

updateAoiTable();
clearSessionData();
resetCalibration();
handleHeatmapStrengthChange();
handleSmoothingChange();
updateFullscreenButtonLabel();
syncCaptureButtons();
syncButtons();
setStatus("Нажмите «Включить камеру»", "Готов к запуску");
