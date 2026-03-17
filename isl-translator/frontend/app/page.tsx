"use client";

import { useEffect, useRef, useState, useCallback } from "react";

/*
 * ISL Translator — Full Frontend
 * 
 * Uses the SIMPLEST proven detection method:
 * - Fingers: TIP.y < PIP.y means finger is up (standard MediaPipe approach)
 * - Thumb: TIP.x vs IP.x comparison (horizontal spread)
 * - Includes live debug panel to show real-time finger states
 */

declare global {
  interface Window {
    Hands: any;
    Camera: any;
    drawConnectors: any;
    drawLandmarks: any;
    HAND_CONNECTIONS: any;
  }
}

// Finger state type for debug display
type FingerState = {
  thumb: boolean;
  index: boolean;
  middle: boolean;
  ring: boolean;
  pinky: boolean;
  count: number;
  raw: string;
};

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [isCameraActive, setIsCameraActive] = useState(false);
  const [gesture, setGesture] = useState<string>("Waiting for gesture...");
  const [targetLanguage, setTargetLanguage] = useState<string>("en-US");
  const [translatedText, setTranslatedText] = useState<string>("");
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [scriptsLoaded, setScriptsLoaded] = useState(false);
  const [statusMessage, setStatusMessage] = useState("Loading MediaPipe...");
  const [fingerDebug, setFingerDebug] = useState<FingerState>({
    thumb: false, index: false, middle: false, ring: false, pinky: false, count: 0, raw: "-"
  });

  const cameraRef = useRef<any>(null);
  const sequenceRef = useRef<number[][]>([]);
  const gestureBufferRef = useRef<string[]>([]);
  const BUFFER_SIZE = 4; // frames before committing
  const lastGestureTimeRef = useRef<number>(0);

  // ─── Load MediaPipe CDN scripts ───────────────────────────────────
  useEffect(() => {
    const scripts = [
      "https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js",
      "https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js",
      "https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js",
    ];

    let loaded = 0;
    scripts.forEach((src) => {
      const script = document.createElement("script");
      script.src = src;
      script.crossOrigin = "anonymous";
      script.onload = () => {
        loaded++;
        if (loaded === scripts.length) {
          setScriptsLoaded(true);
          setStatusMessage("MediaPipe ready. Start camera to begin.");
        }
      };
      script.onerror = () => {
        setStatusMessage("Failed to load MediaPipe. Check your connection.");
      };
      document.head.appendChild(script);
    });
  }, []);

  // ─── MediaPipe results handler ────────────────────────────────────
  // ─── Sequence Helper Functions ──────────────────────────────────
  const extractHandLandmarks = (landmarks: any[]) => {
    // Converts 21 landmarks into a flat array of 63 values (x, y, z)
    return landmarks.flatMap((lm: any) => [lm.x, lm.y, lm.z]);
  };

  const predictGesture = async (sequence: number[][]) => {
    console.log("✅ Sequence completed (30 frames). Calling Model API...");
    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sequence }),
      });
      const data = await response.json();
      if (data.gesture) {
        console.log(`🤖 Model Prediction: ${data.gesture} (Confidence: ${data.confidence})`);
        setGesture(data.gesture);
      }
    } catch (error) {
      console.error("❌ Prediction API Error:", error);
    }
  };

  const onResults = useCallback(
    (results: any) => {
      if (!canvasRef.current) return;
      const canvasCtx = canvasRef.current.getContext("2d");
      if (!canvasCtx) return;

      const w = canvasRef.current.width;
      const h = canvasRef.current.height;

      canvasCtx.save();
      canvasCtx.clearRect(0, 0, w, h);

      // MIRROR the canvas horizontally for natural selfie view
      canvasCtx.translate(w, 0);
      canvasCtx.scale(-1, 1);
      canvasCtx.drawImage(results.image, 0, 0, w, h);

      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        // Only process FIRST hand
        const landmarks = results.multiHandLandmarks[0];
        const handedness = results.multiHandedness?.[0]?.label || "Right";

        if (window.drawConnectors && window.HAND_CONNECTIONS) {
          window.drawConnectors(canvasCtx, landmarks, window.HAND_CONNECTIONS, {
            color: "#00FF88",
            lineWidth: 4,
          });
        }
        if (window.drawLandmarks) {
          window.drawLandmarks(canvasCtx, landmarks, {
            color: "#FF4444",
            lineWidth: 2,
            radius: 4,
          });
        }

        // ── Phase 4 Upgrade: Sequence Processing ──
        const frameLandmarks = extractHandLandmarks(landmarks);
        sequenceRef.current.push(frameLandmarks);

        // Keep track of finger states for the UI debug panel
        updateFingerDebug(landmarks, handedness);

        if (sequenceRef.current.length === 30) {
          predictGesture(sequenceRef.current);
          sequenceRef.current = [];
        }
      }
      canvasCtx.restore();
    },
    []
  );

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // DEBUG HELPER — Separated from classification for the new pipeline
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  const updateFingerDebug = (landmarks: any[], handedness: string) => {
    const tipIds = [4, 8, 12, 16, 20];
    const pipIds = [3, 6, 10, 14, 18];
    const fingers: boolean[] = [];

    if (handedness === "Right") {
      fingers.push(landmarks[tipIds[0]].x < landmarks[pipIds[0]].x);
    } else {
      fingers.push(landmarks[tipIds[0]].x > landmarks[pipIds[0]].x);
    }
    for (let i = 1; i < 5; i++) {
      fingers.push(landmarks[tipIds[i]].y < landmarks[pipIds[i]].y);
    }

    const [thumbUp, indexUp, middleUp, ringUp, pinkyUp] = fingers;
    const fingerCount = fingers.filter(Boolean).length;

    setFingerDebug({
      thumb: thumbUp,
      index: indexUp,
      middle: middleUp,
      ring: ringUp,
      pinky: pinkyUp,
      count: fingerCount,
      raw: `T:${thumbUp ? "↑" : "↓"} I:${indexUp ? "↑" : "↓"} M:${middleUp ? "↑" : "↓"} R:${ringUp ? "↑" : "↓"} P:${pinkyUp ? "↑" : "↓"}`
    });
  };

  // ─── Start / Stop camera ─────────────────────────────────────────
  useEffect(() => {
    if (!isCameraActive || !scriptsLoaded || !videoRef.current) return;

    const hands = new window.Hands({
      locateFile: (file: string) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.5,
    });

    hands.onResults(onResults);

    const cam = new window.Camera(videoRef.current, {
      onFrame: async () => {
        if (videoRef.current) {
          await hands.send({ image: videoRef.current });
        }
      },
      width: 640,
      height: 480,
    });

    cam.start();
    cameraRef.current = cam;
    setStatusMessage("Camera active — show a hand gesture!");

    return () => {
      cam.stop();
      hands.close();
      cameraRef.current = null;
      setStatusMessage("Camera stopped.");
    };
  }, [isCameraActive, scriptsLoaded, onResults]);

  // ─── Translation dictionary ───────────────────────────────────────
  const translations: Record<string, Record<string, string>> = {
    "hi-IN": { 
      HELLO: "नमस्ते", WATER: "पानी", HELP: "मदद", YES: "हाँ", NO: "नहीं",
      I: "मैं", YOU: "आप", EAT: "खाना", DRINK: "पीना", FOOD: "भोजन",
      PLEASE: "कृपया", THANK_YOU: "शुक्रिया", WHERE: "कहाँ", HOSPITAL: "अस्पताल",
      NEED: "ज़रुरत", WANT: "चाहना", GO: "जाना", COME: "आना", MORE: "और", STOP: "रुकिए"
    },
    "mr-IN": { 
      HELLO: "नमस्कार", WATER: "पाणी", HELP: "मदत", YES: "हो", NO: "नाही",
      I: "मी", YOU: "तू", EAT: "जेवण", DRINK: "पिणे", FOOD: "अन्न",
      PLEASE: "कृपया", THANK_YOU: "धन्यवाद", WHERE: "कुठे", HOSPITAL: "रुग्णालय",
      NEED: "गरज", WANT: "पाहिजे", GO: "जाणे", COME: "येणे", MORE: "आणखी", STOP: "थांबा"
    },
    "ta-IN": { 
      HELLO: "வணக்கம்", WATER: "தண்ணீர்", HELP: "உதவி", YES: "ஆம்", NO: "இல்லை",
      I: "நான்", YOU: "நீ", EAT: "சாப்பிடு", DRINK: "குடி", FOOD: "உணவு",
      PLEASE: "தயவுசெய்து", THANK_YOU: "நன்றி", WHERE: "எங்கே", HOSPITAL: "மருத்துவமனை",
      NEED: "தேவை", WANT: "வேண்டும்", GO: "போ", COME: "வா", MORE: "இன்னும்", STOP: "நில்"
    },
    "te-IN": { 
      HELLO: "నమస్కారం", WATER: "నీళ్ళు", HELP: "సహాయం", YES: "అవును", NO: "లేదు",
      I: "నేను", YOU: "నువ్వు", EAT: "తిను", DRINK: "త్రాగు", FOOD: "ఆహారం",
      PLEASE: "దయచేసి", THANK_YOU: "ధన్యవాదాలు", WHERE: "ఎక్కడ", HOSPITAL: "ఆసుపత్రి",
      NEED: "అవసరం", WANT: "కావాలి", GO: "వెళ్ళు", COME: "రా", MORE: "మరింత", STOP: "ఆగు"
    },
    "kn-IN": { 
      HELLO: "ನಮಸ್ಕಾರ", WATER: "ನೀರು", HELP: "ಸಹಾಯ", YES: "ಹೌದು", NO: "ಇಲ್ಲ",
      I: "ನಾನು", YOU: "ನೀವು", EAT: "ಊಟ", DRINK: "ಕುಡಿ", FOOD: "ಆಹಾರ",
      PLEASE: "ದಯವಿಟ್ಟು", THANK_YOU: "ಧನ್ಯವಾದಗಳು", WHERE: "ಎಲ್ಲಿ", HOSPITAL: "ಆಸ್ಪತ್ರೆ",
      NEED: "ಅಗತ್ಯ", WANT: "ಬೇಕು", GO: "ಹೋಗು", COME: "ಬಾ", MORE: "ಇನ್ನೂ", STOP: "ನಿಲ್ಲು"
    },
    "bn-IN": { 
      HELLO: "নমস্কার", WATER: "জল", HELP: "সাহায্য", YES: "হ্যাঁ", NO: "না",
      I: "আমি", YOU: "আপনি", EAT: "খাওয়া", DRINK: "পান করা", FOOD: "খাবার",
      PLEASE: "দয়া করে", THANK_YOU: "ধন্যবাদ", WHERE: "কোথায়", HOSPITAL: "হাসপাতাল",
      NEED: "প্রয়োজন", WANT: "চাই", GO: "যাও", COME: "এসো", MORE: "আরো", STOP: "থামুন"
    },
    "en-US": { 
      HELLO: "Hello", WATER: "Water", HELP: "Help", YES: "Yes", NO: "No",
      I: "I", YOU: "You", EAT: "Eat", DRINK: "Drink", FOOD: "Food",
      PLEASE: "Please", THANK_YOU: "Thank You", WHERE: "Where", HOSPITAL: "Hospital",
      NEED: "Need", WANT: "Want", GO: "Go", COME: "Come", MORE: "More", STOP: "Stop"
    },
  };

  const handleTranslateAndSpeak = useCallback(
    (text: string, lang: string) => {
      const dict = translations[lang] || translations["en-US"];
      const translated = dict[text.toUpperCase()] || text;
      setTranslatedText(translated);
      speakText(translated, lang);
    },
    []
  );

  // Auto-trigger translation when gesture changes
  useEffect(() => {
    if (gesture && gesture !== "Waiting for gesture...") {
      handleTranslateAndSpeak(gesture, targetLanguage);
    }
  }, [gesture, targetLanguage, handleTranslateAndSpeak]);

  // ─── Text-to-Speech ───────────────────────────────────────────────
  const speakText = (text: string, lang: string) => {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    setIsSpeaking(true);
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang;
    utterance.rate = 0.85;
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    window.speechSynthesis.speak(utterance);
  };

  // ─── UI ────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-slate-900 text-white flex flex-col items-center p-4 sm:p-8 font-sans">

      {/* Header */}
      <header className="mb-6 text-center max-w-3xl">
        <div className="inline-block mb-3 px-4 py-1 rounded-full bg-blue-600/20 border border-blue-500/30 text-blue-400 text-sm font-medium tracking-wider uppercase">
          🤟 Real-Time ISL Translator
        </div>
        <h1 className="text-3xl sm:text-5xl font-extrabold mb-3 bg-gradient-to-r from-blue-400 via-cyan-400 to-green-400 text-transparent bg-clip-text">
          Indian Sign Language Translator
        </h1>
        <p className="text-gray-400 text-base sm:text-lg leading-relaxed">
          Show hand gestures → Get text &amp; speech in Hindi, Marathi, Tamil, Telugu &amp; more.
        </p>
      </header>

      {/* Main Content */}
      <main className="flex flex-col lg:flex-row gap-6 w-full max-w-6xl">

        {/* Camera Section */}
        <section className="flex-1 bg-gray-800/60 backdrop-blur-sm rounded-2xl p-5 shadow-2xl border border-gray-700/50">
          <h2 className="text-lg font-semibold mb-3 text-gray-300 flex items-center gap-2">
            <span className="text-2xl">📷</span> Camera Feed
          </h2>

          <div className="relative aspect-video bg-black rounded-xl overflow-hidden flex items-center justify-center border border-gray-700">
            {!isCameraActive && (
              <div className="absolute z-10 text-center p-6 flex flex-col items-center gap-4">
                <div className="text-6xl">🤲</div>
                <p className="text-lg text-gray-300">{statusMessage}</p>
                <button
                  onClick={() => setIsCameraActive(true)}
                  disabled={!scriptsLoaded}
                  className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 disabled:from-gray-600 disabled:to-gray-600 text-white py-3 px-8 rounded-xl text-lg font-semibold transition-all focus:ring-4 focus:ring-blue-400/50 outline-none shadow-lg shadow-blue-600/20"
                  aria-label="Start Camera"
                >
                  {scriptsLoaded ? "▶ Start Camera" : "⏳ Loading..."}
                </button>
              </div>
            )}

            <video ref={videoRef} className="hidden" playsInline />
            <canvas
              ref={canvasRef}
              width={640}
              height={480}
              className={`w-full h-full object-cover ${isCameraActive ? "block" : "hidden"}`}
              aria-label="Live camera feed with hand tracking"
            />
          </div>

          {/* Controls */}
          <div className="mt-4 flex flex-col sm:flex-row justify-between items-center gap-3">
            <button
              onClick={() => setIsCameraActive(!isCameraActive)}
              disabled={!scriptsLoaded}
              className={`py-2.5 px-6 rounded-xl font-semibold transition-all focus:ring-4 outline-none shadow-md ${
                isCameraActive
                  ? "bg-red-600 hover:bg-red-500 focus:ring-red-400/50 shadow-red-600/20"
                  : "bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 focus:ring-blue-400/50 shadow-blue-600/20"
              }`}
            >
              {isCameraActive ? "⏹ Stop Camera" : "▶ Start Camera"}
            </button>

            <div className="flex items-center gap-2">
              <label htmlFor="language" className="font-medium text-gray-400 text-sm">
                🌐 Language:
              </label>
              <select
                id="language"
                value={targetLanguage}
                onChange={(e) => setTargetLanguage(e.target.value)}
                className="bg-gray-700/80 border border-gray-600 rounded-lg p-2 text-white focus:ring-2 focus:ring-cyan-500 outline-none text-sm"
              >
                <option value="en-US">English</option>
                <option value="hi-IN">Hindi (हिंदी)</option>
                <option value="mr-IN">Marathi (मराठी)</option>
                <option value="ta-IN">Tamil (தமிழ்)</option>
                <option value="te-IN">Telugu (తెలుగు)</option>
                <option value="kn-IN">Kannada (ಕನ್ನಡ)</option>
                <option value="bn-IN">Bengali (বাংলা)</option>
              </select>
            </div>
          </div>

          {/* ── LIVE FINGER DEBUG PANEL ── */}
          {isCameraActive && (
            <div className="mt-4 bg-gray-900/80 rounded-xl p-3 border border-gray-700/50">
              <p className="text-xs text-gray-500 mb-2 uppercase tracking-widest font-medium text-center">
                🔍 Live Finger Detection
              </p>
              <div className="flex justify-center gap-3 mb-2">
                {[
                  { name: "👍", label: "Thumb", up: fingerDebug.thumb },
                  { name: "☝️", label: "Index", up: fingerDebug.index },
                  { name: "🖕", label: "Middle", up: fingerDebug.middle },
                  { name: "💍", label: "Ring", up: fingerDebug.ring },
                  { name: "🤙", label: "Pinky", up: fingerDebug.pinky },
                ].map((f) => (
                  <div
                    key={f.label}
                    className={`flex flex-col items-center px-2 py-1 rounded-lg text-xs font-medium transition-colors ${
                      f.up
                        ? "bg-green-600/30 border border-green-500/50 text-green-400"
                        : "bg-red-600/20 border border-red-500/30 text-red-400"
                    }`}
                  >
                    <span className="text-lg">{f.name}</span>
                    <span>{f.label}</span>
                    <span className="font-bold">{f.up ? "UP" : "DOWN"}</span>
                  </div>
                ))}
              </div>
              <p className="text-center text-xs text-gray-400 font-mono">
                Fingers up: {fingerDebug.count} | {fingerDebug.raw}
              </p>
            </div>
          )}

          <p className="mt-3 text-xs text-gray-500 text-center">{statusMessage}</p>
        </section>

        {/* Translation Output Section */}
        <section className="flex-1 bg-gray-800/60 backdrop-blur-sm rounded-2xl p-5 shadow-2xl border border-gray-700/50 flex flex-col">
          <h2 className="text-lg font-semibold mb-4 text-gray-300 flex items-center gap-2">
            <span className="text-2xl">🔊</span> Translation Output
          </h2>

          <div className="flex-1 flex flex-col justify-center gap-5">
            {/* Detected Gesture */}
            <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl p-6 border border-gray-700/60 text-center relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-green-500 to-emerald-500"></div>
              <p className="text-xs text-gray-500 mb-2 uppercase tracking-widest font-medium">
                Detected Gesture
              </p>
              <p className="text-4xl font-black text-green-400 min-h-[48px] tracking-wide">
                {gesture}
              </p>
            </div>

            {/* Translated Text */}
            <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl p-6 border border-gray-700/60 text-center relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 to-cyan-500"></div>
              <p className="text-xs text-gray-500 mb-2 uppercase tracking-widest font-medium">
                Translated Text &amp; Speech
              </p>
              <p className="text-5xl font-black text-cyan-400 min-h-[56px]">
                {translatedText || "—"}
              </p>
              {isSpeaking && (
                <span className="absolute top-4 right-4 flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-cyan-500"></span>
                </span>
              )}
            </div>

            {/* Gesture Guide */}
            <div className="bg-gray-900/50 rounded-xl p-4 border border-gray-700/40">
              <p className="text-xs text-gray-500 mb-3 uppercase tracking-widest font-medium text-center">
                ✋ Gesture Guide
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-sm">
                <div className="bg-gray-800/60 rounded-lg p-2 text-center">
                  <span className="text-xl">🖐️</span>
                  <p className="text-gray-300 font-medium">All 5 Up</p>
                  <p className="text-green-400 text-xs">HELLO</p>
                </div>
                <div className="bg-gray-800/60 rounded-lg p-2 text-center">
                  <span className="text-xl">☝️</span>
                  <p className="text-gray-300 font-medium">Index Only</p>
                  <p className="text-green-400 text-xs">YES</p>
                </div>
                <div className="bg-gray-800/60 rounded-lg p-2 text-center">
                  <span className="text-xl">✌️</span>
                  <p className="text-gray-300 font-medium">Index+Middle</p>
                  <p className="text-green-400 text-xs">WATER</p>
                </div>
                <div className="bg-gray-800/60 rounded-lg p-2 text-center">
                  <span className="text-xl">✊</span>
                  <p className="text-gray-300 font-medium">All Down</p>
                  <p className="text-green-400 text-xs">NO</p>
                </div>
                <div className="bg-gray-800/60 rounded-lg p-2 text-center">
                  <span className="text-xl">🤟</span>
                  <p className="text-gray-300 font-medium">Thumb+Idx+Pinky</p>
                  <p className="text-green-400 text-xs">HELP</p>
                </div>
              </div>
            </div>

            {/* Test Buttons */}
            <div className="flex flex-wrap justify-center gap-2 max-h-48 overflow-y-auto p-2 bg-gray-900/30 rounded-lg">
              {[
                "HELLO", "WATER", "HELP", "YES", "NO",
                "I", "YOU", "EAT", "DRINK", "FOOD",
                "PLEASE", "THANK_YOU", "WHERE", "HOSPITAL",
                "NEED", "WANT", "GO", "COME", "MORE", "STOP"
              ].map((g) => (
                <button
                  key={g}
                  onClick={() => { setGesture(g); handleTranslateAndSpeak(g, targetLanguage); }}
                  className="bg-gray-700/80 hover:bg-gray-600 px-3 py-1.5 rounded-lg text-xs font-medium focus:ring-2 focus:ring-cyan-500/50 outline-none transition-all border border-gray-600/50 hover:border-gray-500"
                >
                  {g}
                </button>
              ))}
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="mt-auto pt-10 pb-4 text-center text-xs text-gray-600 max-w-xl">
        <p className="mb-1">💡 Ensure your environment is well-lit for optimal gesture detection.</p>
        <p>Built with Next.js · MediaPipe Hands · Web Speech API</p>
      </footer>
    </div>
  );
}
