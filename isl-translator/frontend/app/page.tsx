"use client";

import { useEffect, useRef, useState, useCallback } from "react";

// SECTION 1 — MediaPipe CDN globals
declare global {
  interface Window {
    Hands: any;
    Camera: any;
    drawConnectors: any;
    drawLandmarks: any;
    HAND_CONNECTIONS: any;
  }
}

// SECTION 3 — Types and Pure Functions
type Point = { x: number; y: number };
type Stroke = Point[];

const normalizeStrokes = (strokes: Stroke[]): Stroke[] => {
  const allPoints = strokes.flat();
  if (allPoints.length === 0) return strokes;

  const minX = Math.min(...allPoints.map(p => p.x));
  const maxX = Math.max(...allPoints.map(p => p.x));
  const minY = Math.min(...allPoints.map(p => p.y));
  const maxY = Math.max(...allPoints.map(p => p.y));

  const width = maxX - minX;
  const height = maxY - minY;
  const scale = Math.max(width, height, 0.0001);

  return strokes.map(stroke =>
    stroke.map(p => ({
      x: (p.x - (minX + maxX) / 2) / scale + 0.5,
      y: (p.y - (minY + maxY) / 2) / scale + 0.5,
    }))
  );
};

const resamplePath = (points: Point[], n: number): Point[] => {
  if (points.length < 2) return new Array(n).fill(points[0] || { x: 0, y: 0 });
  let totalLength = 0;
  for (let i = 1; i < points.length; i++) {
    totalLength += Math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y);
  }
  const step = totalLength / (n - 1);
  const newPoints: Point[] = [points[0]];
  let accumulated = 0;
  let currentPos = 0;

  for (let i = 1; i < n - 1; i++) {
    const targetDist = i * step;
    while (currentPos < points.length - 1) {
      const d = Math.hypot(points[currentPos + 1].x - points[currentPos].x, points[currentPos + 1].y - points[currentPos].y);
      if (accumulated + d >= targetDist) {
        const t = (targetDist - accumulated) / d;
        newPoints.push({
          x: points[currentPos].x + t * (points[currentPos + 1].x - points[currentPos].x),
          y: points[currentPos].y + t * (points[currentPos + 1].y - points[currentPos].y),
        });
        break;
      }
      accumulated += d;
      currentPos++;
    }
  }
  newPoints.push(points[points.length - 1]);
  return newPoints;
};

// Gesture check helpers
const isPenDown = (lm: any[]) => lm[8].y < lm[6].y && lm[12].y > lm[10].y && lm[16].y > lm[14].y && lm[20].y > lm[18].y;
const isPenUp = (lm: any[]) => lm[8].y < lm[6].y && lm[12].y < lm[10].y && lm[16].y > lm[14].y && lm[20].y > lm[18].y;
const isFist = (lm: any[]) => lm[8].y > lm[6].y && lm[12].y > lm[10].y && lm[16].y > lm[14].y && lm[20].y > lm[18].y;

const matchLetter = (strokes: Stroke[]): string => {
  const allFlat = strokes.flat();
  if (allFlat.length < 4) return "?";

  const norm = normalizeStrokes(strokes);
  const flatNorm = norm.flat();
  const numStrokes = strokes.length;

  const minX = Math.min(...flatNorm.map(p => p.x));
  const maxX = Math.max(...flatNorm.map(p => p.x));
  const minY = Math.min(...flatNorm.map(p => p.y));
  const maxY = Math.max(...flatNorm.map(p => p.y));
  const w = maxX - minX;
  const h = maxY - minY;
  const aspect = w / (h || 1);
  const isTall = aspect < 0.75;
  const isWide = aspect > 1.25;

  const s = norm.map(stroke => resamplePath(stroke, 16));
  const s0 = s[0] || [];
  const s1 = s[1] || [];
  const s2 = s[2] || [];

  const strokeIsHoriz = (st: Point[]) => Math.abs(st[15].x - st[0].x) > Math.abs(st[15].y - st[0].y) * 1.5;
  const strokeIsVert = (st: Point[]) => Math.abs(st[15].y - st[0].y) > Math.abs(st[15].x - st[0].x) * 1.5;
  const strokeGoesDown = (st: Point[]) => st[15].y > st[0].y + 0.2;
  const strokeGoesRight = (st: Point[]) => st[15].x > st[0].x + 0.2;
  const strokeGoesLeft = (st: Point[]) => st[15].x < st[0].x - 0.2;
  const strokeLen = (st: Point[]) => {
    let l = 0;
    for (let i = 1; i < st.length; i++) l += Math.hypot(st[i].x - st[i - 1].x, st[i].y - st[i - 1].y);
    return l;
  };

  const closed0 = s0.length > 0 ? Math.hypot(s0[15].x - s0[0].x, s0[15].y - s0[0].y) : 1;

  if (numStrokes === 1) {
    let dirChanges = 0;
    for (let i = 1; i < s0.length - 1; i++) {
        const v1 = { x: s0[i].x - s0[i - 1].x, y: s0[i].y - s0[i - 1].y };
        const v2 = { x: s0[i + 1].x - s0[i].x, y: s0[i + 1].y - s0[i].y };
        const dot = v1.x * v2.x + v1.y * v2.y;
        if (dot < -0.002) dirChanges++;
    }
    const topPts = s0.filter(p => p.y < 0.5).length / 16;
    const bottomPts = s0.filter(p => p.y > 0.5).length / 16;
    const rightPts = s0.filter(p => p.x > 0.5).length / 16;
    const mid = s0[8];
    const isClosed = closed0 < 0.22;
    const pLen = strokeLen(s0);
    const start = s0[0], end = s0[15];

    const goesDown = end.y > start.y + 0.25;
    const goesUp = end.y < start.y - 0.25;
    const goesRight = end.x > start.x + 0.25;
    const goesLeft = end.x < start.x - 0.25;

    let crossV = false, crossH = false;
    for (let i = 1; i < s0.length; i++) {
      if ((s0[i].x < 0.5 && s0[i-1].x > 0.5) || (s0[i].x > 0.5 && s0[i-1].x < 0.5)) crossV = true;
      if ((s0[i].y < 0.5 && s0[i-1].y > 0.5) || (s0[i].y > 0.5 && s0[i-1].y < 0.5)) crossH = true;
    }

    // Single Stroke B (Tall, multiple loops on the right)
    if (isTall && dirChanges >= 3 && rightPts > 0.4 && pLen > 1.4 && !isClosed) return "B";
    if (isTall && dirChanges >= 2 && goesRight && rightPts > 0.4 && start.y > 0.3 && pLen > 1.2) return "R";
    if (isTall && dirChanges === 2 && rightPts > 0.3 && !crossV && !crossH) return "K";

    if (isClosed && dirChanges >= 3 && pLen > 1.2) return "O";
    if (!isClosed && dirChanges <= 2 && start.x > 0.4 && end.x > 0.4 && start.y < 0.5 && end.y > 0.5) return "C";
    if (start.y < 0.4 && end.y < 0.4 && bottomPts > 0.25 && Math.abs(start.x - end.x) < 0.55) return "U";
    if (mid.y > start.y + 0.1 && mid.y > end.y + 0.1 && start.x < end.x && dirChanges <= 2) return "V";
    if (dirChanges === 1 && (goesDown || start.y < 0.4) && (goesRight || end.x > 0.6)) return "L";
    if (strokeIsVert(s0) && dirChanges <= 1) return "I";
    if (dirChanges === 2 && start.y < 0.4 && end.y > 0.6) return "Z";
    if (dirChanges === 2 && pLen > 1.3 && isTall) return "S";
    if (dirChanges === 2 && start.y < 0.35 && end.y > 0.6) return "N";
    if (dirChanges >= 3 && start.y < 0.35 && end.y < 0.45) return "M";
    if (isWide && dirChanges >= 3 && bottomPts > 0.3) return "W";
    if (dirChanges === 1 && closed0 < 0.4 && rightPts > 0.35) return "D";
    if (topPts > 0.35 && mid.y < 0.5 && !isClosed && dirChanges === 1) return "A";
    if (dirChanges >= 2 && goesDown && topPts > 0.35) return "P";
    if (!isClosed && dirChanges <= 2 && end.y > 0.3 && end.y < 0.7 && start.x > 0.3) return "G";
    if (goesDown && goesLeft && dirChanges === 1) return "J";
    if (dirChanges === 1 && crossV && crossH) return "X";
    if (dirChanges === 2 && start.y < 0.4 && mid.y > 0.5 && end.y < 0.4) return "Y";
    if (dirChanges >= 2 && closed0 < 0.4 && goesDown) return "Q";
    if (strokeIsVert(s0)) return "I";
    return "?";
  }

  if (numStrokes === 2) {
    const h0 = strokeIsHoriz(s0), h1 = strokeIsHoriz(s1), v0 = strokeIsVert(s0), v1 = strokeIsVert(s1);
    const diag0 = !h0 && !v0, diag1 = !h1 && !v1;
    const start1 = s1[0], end1 = s1[15];

    if (h0 && v1 && strokeGoesDown(s1) && start1.y < 0.4) return "T";
    if (v0 && h1 && end1.y > 0.3 && start1.y < 0.5) return "F";
    if (v0 && h1 && start1.y > 0.6) return "L";
    if (v0 && !h1 && !v1) return "K";
    if (v0 && strokeGoesRight(s1) && start1.y > 0.3 && start1.y < 0.6) return "R";
    if (diag0 && diag1 && (s0[15].y > 0.6 || s1[15].y > 0.6) && strokeGoesDown(s0)) return "V";
    if (diag0 && diag1) return "X";
    if (diag0 && v1 && strokeGoesDown(s1) && start1.y > 0.4) return "Y";
    return "?";
  }

  if (numStrokes === 3) {
    const h0 = strokeIsHoriz(s0), h1 = strokeIsHoriz(s1), h2 = strokeIsHoriz(s2);
    const v0 = strokeIsVert(s0), v1 = strokeIsVert(s1), v2 = strokeIsVert(s2);

    if ((h0 && h1 && h2) || (v0 && h1 && h2)) return "E";
    if (v0 && h1 && h2 && s2[15].y < 0.6) return "F";
    if (v0 && v2) return "H";
    if (!v0 && !v1 && h2) return "A";
    if (v0 && !h1 && v2) return "N";
    if (h0 && strokeGoesDown(s1) && h2) return "Z";
    if (v0) return "K";
    return "?";
  }

  if (numStrokes >= 4) {
      if (isWide) return "W";
      return "E";
  }

  return "?";
};

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cameraRef = useRef<any>(null);

  // SECTION 4 — State
  const strokesRef = useRef<Stroke[]>([]);
  const currentStrokeRef = useRef<Stroke>([]);
  const penDownRef = useRef(false);
  const hasDrawnRef = useRef(false);
  const fistTimerRef = useRef<NodeJS.Timeout | null>(null);

  const [scriptsLoaded, setScriptsLoaded] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [statusMessage, setStatusMessage] = useState("Loading MediaPipe...");
  const [currentLetter, setCurrentLetter] = useState<string>("?");
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [letterHistory, setLetterHistory] = useState<string[]>([]);
  const [penMode, setPenMode] = useState<"up" | "down" | "idle">("idle");
  const [phase, setPhase] = useState<"idle" | "drawing" | "recognized">("idle");
  const [strokeCount, setStrokeCount] = useState(0);

  // SECTION 1 (cont.) — Load Scripts
  useEffect(() => {
    const scripts = [
      "https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js",
      "https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js",
      "https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js",
    ];
    let loaded = 0;
    scripts.forEach(src => {
      const s = document.createElement("script");
      s.src = src;
      s.crossOrigin = "anonymous";
      s.onload = () => {
        loaded++;
        if (loaded === scripts.length) setScriptsLoaded(true);
      };
      document.head.appendChild(s);
    });
  }, []);

  // SECTION 5 — Functions
  const speak = useCallback((text: string) => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "en-US";
    u.rate = 0.85;
    u.pitch = 1.1;
    u.onstart = () => setIsSpeaking(true);
    u.onend = () => setIsSpeaking(false);
    u.onerror = () => setIsSpeaking(false);
    window.speechSynthesis.speak(u);
  }, []);

  const recognizeNow = useCallback(() => {
    if (currentStrokeRef.current.length > 3) {
      strokesRef.current.push([...currentStrokeRef.current]);
      currentStrokeRef.current = [];
    }
    penDownRef.current = false;
    const flatAll = strokesRef.current.flat();
    if (flatAll.length < 4) {
      strokesRef.current = [];
      currentStrokeRef.current = [];
      hasDrawnRef.current = false;
      setStatusMessage("Draw something first!");
      setStrokeCount(0);
      return;
    }

    const letter = matchLetter(strokesRef.current);
    strokesRef.current = [];
    currentStrokeRef.current = [];
    hasDrawnRef.current = false;
    setCurrentLetter(letter);
    setPhase("recognized");
    setStrokeCount(0);

    if (letter !== "?") {
      speak(letter);
      setLetterHistory(prev => [...prev.slice(-19), letter]);
      setStatusMessage("✓ Got it: " + letter);
    } else {
      setStatusMessage("Couldn't read that — try again!");
    }

    setTimeout(() => setPhase("idle"), 2500);
  }, [speak]);

  const drawAllStrokes = useCallback((ctx: CanvasRenderingContext2D, W: number, H: number) => {
    const drawStroke = (stroke: Stroke, color: string, alpha: number) => {
      if (stroke.length < 2) return;
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = 6;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.shadowColor = color;
      ctx.shadowBlur = 12;
      ctx.moveTo((1 - stroke[0].x) * W, stroke[0].y * H);
      for (let i = 1; i < stroke.length; i++) {
        ctx.lineTo((1 - stroke[i].x) * W, stroke[i].y * H);
      }
      ctx.stroke();
    };

    strokesRef.current.forEach(s => drawStroke(s, "#00FFAA", 0.85));
    drawStroke(currentStrokeRef.current, "#00FFAA", 1.0);
  }, []);

  const clearAll = () => {
    strokesRef.current = [];
    currentStrokeRef.current = [];
    penDownRef.current = false;
    hasDrawnRef.current = false;
    setStrokeCount(0);
    setPhase("idle");
    setCurrentLetter("?");
    setStatusMessage("System ready. ☝️ Start drawing.");
  };

  const onResults = useCallback((results: any) => {
    if (!canvasRef.current || !videoRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;

    const W = canvasRef.current.width;
    const H = canvasRef.current.height;

    ctx.save();
    ctx.clearRect(0, 0, W, H);
    ctx.translate(W, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(results.image, 0, 0, W, H);

    if (results.multiHandLandmarks?.length > 0) {
      const lm = results.multiHandLandmarks[0];
      if (window.drawConnectors && window.HAND_CONNECTIONS) {
        window.drawConnectors(ctx, lm, window.HAND_CONNECTIONS, { color: "#ffffff15", lineWidth: 2 });
        window.drawLandmarks(ctx, lm, { color: "#ffffff25", radius: 2 });
      }

      const isD = isPenDown(lm);
      const isU = isPenUp(lm);
      const isF = isFist(lm);

      if (isD) {
        setPenMode("down");
        penDownRef.current = true;
        hasDrawnRef.current = true;
        if (fistTimerRef.current) clearTimeout(fistTimerRef.current);
        currentStrokeRef.current.push({ x: 1 - lm[8].x, y: lm[8].y });
        setPhase("drawing");
        setStatusMessage("✍️ Drawing... ✌️ 2 fingers = lift pen, ✊ fist = done");
        
        // Fingertip dot
        ctx.beginPath();
        ctx.fillStyle = "#00FFAA";
        ctx.shadowBlur = 25;
        ctx.arc(lm[8].x * W, lm[8].y * H, 10, 0, Math.PI * 2);
        ctx.fill();
      } else if (isU && penDownRef.current) {
        setPenMode("up");
        penDownRef.current = false;
        if (currentStrokeRef.current.length > 3) {
          strokesRef.current.push([...currentStrokeRef.current]);
          currentStrokeRef.current = [];
          setStrokeCount(strokesRef.current.length);
        }
        setStatusMessage(`✌️ Pen lifted — ${strokesRef.current.length} stroke(s) saved. ☝️ 1 finger to continue, ✊ fist = done`);
      } else if (isF && hasDrawnRef.current) {
        setPenMode("idle");
        hasDrawnRef.current = false;
        if (!fistTimerRef.current) {
          fistTimerRef.current = setTimeout(() => {
            recognizeNow();
            fistTimerRef.current = null;
          }, 200);
        }
      } else {
        setPenMode("idle");
        if (phase === "idle" && !hasDrawnRef.current) {
            setStatusMessage("☝️ 1 finger = draw  ✌️ 2 fingers = lift  ✊ fist = done");
        }
      }
    }

    drawAllStrokes(ctx, W, H);
    ctx.restore();
  }, [recognizeNow, drawAllStrokes, phase]);

  // SECTION 1 (cont.) — Camera
  useEffect(() => {
    if (!isCameraActive || !scriptsLoaded || !videoRef.current) return;
    const hands = new window.Hands({
      locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });
    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.75,
      minTrackingConfidence: 0.6
    });
    hands.onResults(onResults);
    const cam = new window.Camera(videoRef.current, {
      onFrame: async () => { if (videoRef.current) await hands.send({ image: videoRef.current }); },
      width: 640,
      height: 480
    });
    cam.start();
    return () => { cam.stop(); hands.close(); };
  }, [isCameraActive, scriptsLoaded, onResults]);

  // SECTION 6 — Full UI
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-slate-900 text-white font-sans selection:bg-emerald-500/30">
      
      {/* Header */}
      <header className="p-8 md:p-12 text-center animate-in fade-in slide-in-from-top-4 duration-1000">
        <div className="inline-block mb-4 px-4 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-black uppercase tracking-[0.2em] shadow-xl">
           ✍️ Air Writing — Multi-Stroke
        </div>
        <h1 className="text-4xl md:text-7xl font-black mb-4 tracking-tighter bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
          Draw Letters in Air
        </h1>
        <p className="text-gray-400 text-lg md:text-xl font-medium max-w-2xl mx-auto">
          One finger draws <span className="text-white">·</span> Two fingers lifts the pen <span className="text-white">·</span> Fist means done
        </p>
      </header>

      {/* Gesture Guide Row */}
      <div className="max-w-7xl mx-auto px-4 grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
        <div className="bg-emerald-500/5 border border-emerald-500/10 p-6 rounded-3xl flex flex-col items-center text-center">
            <span className="text-5xl mb-3">☝️</span>
            <h3 className="text-lg font-black text-emerald-400 uppercase tracking-widest mb-1">1 finger</h3>
            <p className="text-sm text-gray-500 font-bold">Pen DOWN — fingertip is tracked</p>
        </div>
        <div className="bg-yellow-500/5 border border-yellow-500/10 p-6 rounded-3xl flex flex-col items-center text-center">
            <span className="text-5xl mb-3">✌️</span>
            <h3 className="text-lg font-black text-yellow-500 uppercase tracking-widest mb-1">2 fingers</h3>
            <p className="text-sm text-gray-500 font-bold">Pen UP — lift and reposition</p>
        </div>
        <div className="bg-blue-500/5 border border-blue-500/10 p-6 rounded-3xl flex flex-col items-center text-center">
            <span className="text-5xl mb-3">✊</span>
            <h3 className="text-lg font-black text-blue-500 uppercase tracking-widest mb-1">Fist</h3>
            <p className="text-sm text-gray-500 font-bold">DONE — trigger recognition</p>
        </div>
      </div>

      <main className="max-w-7xl mx-auto px-4 md:px-8 pb-12 flex flex-col lg:flex-row gap-8">
        
        {/* Left Panel */}
        <div className="flex-1 flex flex-col gap-6">
          <div className="bg-gray-800/60 backdrop-blur-2xl rounded-3xl border border-gray-700/50 p-6 shadow-2xl overflow-hidden">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-black uppercase text-gray-400 flex items-center gap-2">
                <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" /> 📷 Draw Here
              </h2>
              <div className="flex items-center gap-4">
                 {strokeCount > 0 && (
                   <span className="px-3 py-1 rounded-lg bg-gray-900 border border-gray-700 text-[10px] font-black text-gray-400 uppercase tracking-widest">
                     {strokeCount} STROKE(S)
                   </span>
                 )}
                 <div className={`px-4 py-1.5 rounded-full flex items-center gap-2 text-[10px] font-black uppercase tracking-widest ${
                   penMode === "down" ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30" :
                   penMode === "up" ? "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30" :
                   "bg-gray-700/50 text-gray-500 border border-gray-600/30"
                 }`}>
                   <span className={`w-2 h-2 rounded-full ${
                     penMode === "down" ? "bg-emerald-500 animate-ping" : 
                     penMode === "up" ? "bg-yellow-500" : "bg-gray-500"
                   }`} />
                   {penMode === "down" ? "✍️ Pen DOWN" : penMode === "up" ? "✌️ Pen LIFTED" : "● Idle"}
                 </div>
              </div>
            </div>

            <div className="aspect-video bg-black rounded-2xl relative flex items-center justify-center overflow-hidden">
              {!isCameraActive ? (
                <div className="flex flex-col items-center gap-6 p-8 text-center">
                  <span className="text-[120px] drop-shadow-2xl opacity-80">✍️</span>
                  <p className="text-lg text-gray-500 font-medium">{statusMessage}</p>
                  <button
                    onClick={() => setIsCameraActive(true)}
                    disabled={!scriptsLoaded}
                    className="px-12 py-4 bg-white text-black font-black rounded-2xl transition-all hover:scale-110 active:scale-95 shadow-2xl disabled:opacity-50"
                  >
                    START CAMERA
                  </button>
                </div>
              ) : (
                <>
                  <video ref={videoRef} className="hidden" />
                  <canvas ref={canvasRef} width={640} height={480} className="w-full h-full object-cover" />
                  <div className={`absolute bottom-0 inset-x-0 h-1.5 transition-colors duration-500 ${
                    phase === "drawing" ? "bg-emerald-500 shadow-[0_0_15px_#10b981]" : 
                    phase === "recognized" ? "bg-cyan-500 shadow-[0_0_15px_#06b6d4]" : "bg-gray-700"
                  }`} />
                </>
              )}
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              <button
                onClick={() => setIsCameraActive(!isCameraActive)}
                className={`flex-1 py-3 font-black rounded-xl transition-all uppercase tracking-widest text-[11px] ${
                  isCameraActive ? "bg-red-500/10 text-red-500 border border-red-500/20" : "bg-emerald-500 text-black"
                }`}
              >
                {isCameraActive ? "■ Stop Camera" : "▶ Start Camera"}
              </button>
              <button onClick={clearAll} className="flex-1 py-3 bg-gray-700 text-white font-black rounded-xl border border-gray-600 transition-all uppercase tracking-widest text-[11px]">
                  🗑 Clear All
              </button>
              <button onClick={recognizeNow} className="flex-1 py-3 bg-cyan-600 text-white font-black rounded-xl shadow-lg transition-all uppercase tracking-widest text-[11px]">
                  ✊ Recognize Now
              </button>
            </div>
            <p className="mt-4 text-[10px] text-gray-500 text-center font-black uppercase tracking-widest opacity-60">Status: {statusMessage}</p>
          </div>

          <div className="flex flex-col md:flex-row gap-6">
            <div className="flex-1 bg-black/40 backdrop-blur-xl rounded-3xl border border-white/5 p-10 flex flex-col items-center justify-center min-h-[220px]">
               <p className="text-[10px] text-gray-500 font-black uppercase tracking-widest mb-2">Detected</p>
               <div className={`text-9xl font-black transition-all duration-500 ${phase === "recognized" ? "text-emerald-400 scale-110" : "text-gray-800"}`}>
                 {currentLetter}
               </div>
               {isSpeaking && (
                 <div className="mt-4 flex flex-col items-center gap-2">
                    <div className="flex gap-1 items-end h-6">
                      {[0.1, 0.3, 0.2, 0.4].map((d, i) => (
                        <div key={i} className="w-1 bg-cyan-400 rounded-full animate-bounce" style={{ height: '100%', animationDelay: i * 0.1 + 's' }} />
                      ))}
                    </div>
                    <span className="text-[10px] text-cyan-400 font-black uppercase tracking-widest">Speaking "{currentLetter}"</span>
                 </div>
               )}
            </div>

            <div className="flex-[1.5] bg-black/40 backdrop-blur-xl rounded-3xl border border-white/5 p-8 flex flex-col">
               <div className="flex items-center justify-between mb-4">
                 <h3 className="text-[10px] font-black text-gray-500 uppercase tracking-widest">History</h3>
                 <button onClick={() => setLetterHistory([])} className="text-[10px] font-black text-gray-500 hover:text-white uppercase">Clear</button>
               </div>
               <div className="flex flex-wrap gap-2 mb-6 min-h-[48px]">
                 {letterHistory.map((l, i) => (
                   <span key={i} className="w-10 h-10 flex items-center justify-center bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-black rounded-xl">{l}</span>
                 ))}
               </div>
               <div className="mt-auto pt-6 border-t border-gray-800">
                  <p className="text-[10px] text-gray-500 mb-1 font-black uppercase">Word</p>
                  <p className="text-3xl font-black text-cyan-300 tracking-widest">{letterHistory.join("") || "—"}</p>
               </div>
            </div>
          </div>
        </div>

        {/* Right Panel */}
        <aside className="lg:w-80 flex flex-col gap-6">
          <div className="bg-gray-800/60 backdrop-blur-xl rounded-3xl border border-gray-700/50 p-6 flex flex-col shadow-2xl h-full">
            <h2 className="text-sm font-black mb-6 uppercase tracking-widest flex items-center gap-2">
               <span className="w-1.5 h-6 bg-emerald-500 rounded-full" /> How to Draw
            </h2>
            <div className="overflow-y-auto custom-scrollbar space-y-2 max-h-[500px] pr-2">
              {[
                { l: "A", h: "Two diagonals ↗↘ + crossbar —" }, { l: "B", h: "Vertical ↓ + two bumps ))" },
                { l: "C", h: "One arc open right ⌒" }, { l: "D", h: "Vertical ↓ + right curve" },
                { l: "E", h: "Vertical ↓ + 3 horizontal lines" }, { l: "F", h: "Vertical ↓ + 2 horizontal lines at top" },
                { l: "G", h: "C arc that hooks inward" }, { l: "H", h: "↓ crossbar ↓ three strokes" },
                { l: "I", h: "Just straight ↓" }, { l: "J", h: "↓ then hook left ⌐" },
                { l: "K", h: "↓ then two diagonals <>" }, { l: "L", h: "↓ then → at bottom" },
                { l: "M", h: "↓↗↓↗ zigzag" }, { l: "N", h: "↓ diagonal ↓ three strokes" },
                { l: "O", h: "Full circle ○" }, { l: "P", h: "↓ + upper right bump" },
                { l: "Q", h: "○ circle + tail ↘" }, { l: "R", h: "↓ + bump + diagonal leg" },
                { l: "S", h: "One wavy S stroke" }, { l: "T", h: "— horizontal then ↓ vertical" },
                { l: "U", h: "↓ curve back ↑" }, { l: "V", h: "↘ then ↗" },
                { l: "W", h: "↘↗↘↗ four strokes" }, { l: "X", h: "↘ diagonal + ↙ diagonal" },
                { l: "Y", h: "↘ + ↙ + ↓ stem" }, { l: "Z", h: "→ diagonal ↙ then →" }
              ].map(it => (
                <div key={it.l} className="flex items-center gap-3 p-2 rounded-xl bg-gray-900/50 border border-white/5">
                   <span className="w-8 h-8 flex items-center justify-center bg-emerald-500 text-black font-black rounded-lg text-xs">{it.l}</span>
                   <span className="text-[10px] font-bold text-gray-400 uppercase tracking-tight">{it.h}</span>
                </div>
              ))}
            </div>

            <div className="mt-8">
               <p className="text-[10px] font-black text-gray-500 uppercase tracking-widest mb-4">Tap to hear</p>
               <div className="flex flex-wrap gap-1">
                 {"ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("").map(l => (
                   <button key={l} onClick={() => { setCurrentLetter(l); speak(l); setLetterHistory(prev => [...prev.slice(-19), l]); }} className="w-7 h-7 flex items-center justify-center bg-gray-950 border border-gray-800 text-[9px] font-black text-gray-500 hover:text-white rounded-lg transition-all">{l}</button>
                 ))}
               </div>
            </div>

            <div className="mt-8 bg-yellow-500/5 border border-yellow-500/10 rounded-2xl p-5">
               <h4 className="text-yellow-500 text-[10px] font-black uppercase tracking-widest mb-3 flex items-center gap-2">💡 Tips</h4>
               <ul className="text-[9px] text-yellow-500/70 font-bold space-y-2 uppercase leading-relaxed">
                  <li>☝️ one finger = drawing</li>
                  <li>✌️ two fingers = lifted</li>
                  <li>✊ fist = RECOGNIZE</li>
                  <li>Draw BIG and slow</li>
                  <li>Use "Recognize Now" if needed</li>
               </ul>
            </div>
          </div>
        </aside>
      </main>

      <style jsx global>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(16, 185, 129, 0.2); border-radius: 2px; }
      `}</style>
    </div>
  );
}
