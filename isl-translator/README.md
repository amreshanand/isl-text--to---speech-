# 🤟 Indian Sign Language (ISL) Translator

> A real-time accessibility platform that translates **Indian Sign Language hand gestures** into **text and speech** across 7 languages. Built to help deaf and speech-impaired individuals communicate without a human interpreter.

---

## 📋 Table of Contents

- [Quick Start (TL;DR)](#-quick-start-tldr)
- [Prerequisites](#-prerequisites)
- [Project Structure](#-project-structure)
- [Step-by-Step Setup](#-step-by-step-setup)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Backend Setup (Python/FastAPI)](#2-backend-setup-pythonfastapi)
  - [3. Frontend Setup (Next.js)](#3-frontend-setup-nextjs)
  - [4. AI Model Training (Optional)](#4-ai-model-training-optional)
- [Running the Project](#-running-the-project)
- [How It Works](#-how-it-works)
- [Supported Gestures](#-supported-gestures)
- [Supported Languages](#-supported-languages)
- [Troubleshooting](#-troubleshooting)
- [Deployment](#-deployment)
- [Tech Stack](#-tech-stack)

---

## ⚡ Quick Start (TL;DR)

> **Need it running in 2 minutes?** Open **two terminal windows** and run:

### Terminal 1 — Backend

```bash
cd isl-translator/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2 — Frontend

```bash
cd isl-translator/frontend
npm install
npm run dev
```

### Then Open

```
🌐 Frontend:  http://localhost:3000
🔧 Backend:   http://localhost:8000
📖 API Docs:  http://localhost:8000/docs
```

---

## ✅ Prerequisites

Make sure you have the following installed **before starting**:

| Tool       | Required Version | Check Command         | Install Link                                           |
| ---------- | ---------------- | --------------------- | ------------------------------------------------------ |
| **Node.js** | v18 or later    | `node --version`      | [nodejs.org](https://nodejs.org/)                      |
| **npm**     | v8 or later     | `npm --version`       | Comes with Node.js                                     |
| **Python**  | v3.9 or later   | `python3 --version`   | [python.org](https://www.python.org/downloads/)        |
| **pip**     | Latest          | `pip --version`       | Comes with Python                                      |
| **Git**     | Any             | `git --version`       | [git-scm.com](https://git-scm.com/)                   |
| **Webcam**  | Any USB/built-in| —                     | Required for gesture detection                         |

> 💡 **macOS users**: Python is usually pre-installed. Use `python3` and `pip3` instead of `python` and `pip`.

---

## 📁 Project Structure

```
isl-text--to---speech-/
└── isl-translator/                 ← Root project folder
    ├── README.md                   ← You are here
    │
    ├── frontend/                   ← 🌐 Next.js Web Application
    │   ├── app/
    │   │   ├── page.tsx            ← Main page (camera + translator UI)
    │   │   ├── layout.tsx          ← Root layout
    │   │   ├── globals.css         ← Global styles
    │   │   └── favicon.ico         ← Favicon
    │   ├── public/                 ← Static assets
    │   ├── package.json            ← Frontend dependencies
    │   ├── next.config.ts          ← Next.js configuration
    │   ├── tsconfig.json           ← TypeScript configuration
    │   └── node_modules/           ← (auto-generated after npm install)
    │
    ├── backend/                    ← 🐍 FastAPI Python Server
    │   ├── main.py                 ← API endpoints (/predict, /users, /stats)
    │   ├── requirements.txt        ← Python dependencies
    │   └── venv/                   ← (auto-generated after creating virtualenv)
    │
    └── ai-model/                   ← 🤖 AI Model Training Scripts
        ├── collect_data.py         ← Record hand gesture dataset via webcam
        ├── train.py                ← Train Dense neural network model
        └── train_lstm.py           ← Train LSTM sequence model (recommended)
```

---

## 🔧 Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/isl-text--to---speech-.git
cd isl-text--to---speech-/isl-translator
```

---

### 2. Backend Setup (Python/FastAPI)

```bash
# Step 1: Navigate to the backend folder
cd backend

# Step 2: Create a Python virtual environment
python3 -m venv venv

# Step 3: Activate the virtual environment
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows (use this instead)

# Step 4: Install Python dependencies
pip install -r requirements.txt
```

**📦 What gets installed:**

| Package              | Purpose                              |
| -------------------- | ------------------------------------ |
| `fastapi`            | Web framework for the API server     |
| `uvicorn`            | ASGI server to run FastAPI           |
| `pydantic`           | Data validation for API requests     |
| `numpy`              | Numerical processing for landmarks   |
| `opencv-python`      | Computer vision utilities            |
| `mediapipe`          | Hand landmark detection              |
| `tensorflow`         | Deep learning model inference        |
| `scikit-learn`       | Model training utilities             |

> ⚠️ **Note:** `tensorflow` is a large package (~500 MB+). The first install may take several minutes.

---

### 3. Frontend Setup (Next.js)

Open a **new terminal window** (keep the backend terminal open):

```bash
# Step 1: Navigate to the frontend folder
cd isl-translator/frontend

# Step 2: Install Node.js dependencies
npm install
```

**📦 Key frontend dependencies:**

| Package              | Purpose                                      |
| -------------------- | -------------------------------------------- |
| `next`               | React framework (App Router)                 |
| `react` / `react-dom`| UI rendering                                 |
| `@mediapipe/hands`   | Hand detection via CDN scripts               |
| `@tensorflow/tfjs`   | TensorFlow.js for browser-side inference     |
| `axios`              | HTTP client for API calls                    |
| `tailwindcss`        | Utility-first CSS framework                  |

---

### 4. AI Model Training (Optional)

> 🔔 **You can skip this step!** The backend runs with **mock predictions** by default, so no trained model is needed to test the app.

If you want to train your own gesture recognition model:

#### Step A — Collect Gesture Data

```bash
cd ai-model
python3 collect_data.py
```

- Your **webcam will open** showing a live feed
- Press **`s`** to start recording each gesture sequence
- Press **`q`** to quit at any time
- Data saves as `.npy` files in the `dataset/` folder
- Records **20 sequences × 30 frames** for each of the 20 gestures

#### Step B — Train the Model

**Option 1: Dense Neural Network** (simpler, faster)

```bash
python3 train.py
```

**Option 2: LSTM Network** (recommended, better accuracy)

```bash
python3 train_lstm.py
```

#### Step C — Use the Trained Model

After training, uncomment the model loading lines in `backend/main.py` (lines 33–34):

```python
# Change from:
# import tensorflow as tf
# model = tf.keras.models.load_model('../ai-model/saved_model/isl_model.h5')

# Change to:
import tensorflow as tf
model = tf.keras.models.load_model('../ai-model/saved_model/isl_model.h5')
```

---

## 🚀 Running the Project

> **You need TWO terminal windows running simultaneously.**

### Terminal 1 ➜ Start the Backend API

```bash
cd isl-translator/backend
source venv/bin/activate          # Activate virtualenv every time
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

✅ You should see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
Model loaded successfully (Placeholder).
```

### Terminal 2 ➜ Start the Frontend

```bash
cd isl-translator/frontend
npm run dev
```

✅ You should see:

```
  ▲ Next.js
  - Local:   http://localhost:3000
  ✓ Ready
```

### Open the App

1. Open **http://localhost:3000** in your web browser (Chrome recommended)
2. Click **"▶ Start Camera"**
3. **Allow camera access** when prompted
4. Show hand gestures to get real-time text and speech output!

---

## 🧠 How It Works

```
┌──────────────┐    ┌───────────────────┐    ┌──────────────────┐
│   📷 Webcam   │───▶│  MediaPipe Hands   │───▶│  21 Hand Points  │
│  (Browser)    │    │  (CDN scripts)     │    │  (63 values)     │
└──────────────┘    └───────────────────┘    └────────┬─────────┘
                                                      │
                                            30 frames collected
                                                      │
                                                      ▼
┌──────────────┐    ┌───────────────────┐    ┌──────────────────┐
│ 🔊 Speaker    │◀───│  Translation +     │◀───│  FastAPI Backend  │
│ (Browser TTS) │    │  Text-to-Speech    │    │  POST /predict   │
└──────────────┘    └───────────────────┘    └──────────────────┘
```

**Data Flow:**

1. **Camera** captures live video frames in the browser
2. **MediaPipe Hands** detects 21 hand landmarks (x, y, z) per frame = 63 values
3. **Frontend** collects 30 consecutive frames into a sequence
4. **Sequence is sent** via `POST /predict` to the FastAPI backend
5. **Backend** processes the landmarks and returns the predicted gesture
6. **Translation** converts the gesture label to the selected language
7. **Text-to-Speech** speaks the translated word out loud

---

## ✋ Supported Gestures

| Gesture       | Hand Position                | Emoji |
| ------------- | ---------------------------- | ----- |
| **HELLO**     | All 5 fingers up (open palm) | 🖐️    |
| **YES**       | Only index finger up         | ☝️    |
| **NO**        | All fingers down (fist)      | ✊    |
| **WATER**     | Index + middle fingers up    | ✌️    |
| **HELP**      | Thumb + index + pinky up     | 🤟    |
| **I**         | Pinky up only                | 🤙    |
| **YOU**       | Index pointing               | 👉    |
| **EAT**       | Fingers to mouth motion      | 🍽️    |
| **DRINK**     | Thumb up gesture             | 🥤    |
| **FOOD**      | Combined eating gesture      | 🍕    |
| **PLEASE**    | Prayer hands                 | 🙏    |
| **THANK_YOU** | Gratitude motion             | 🤝    |
| **WHERE**     | Question pose                | ❓    |
| **HOSPITAL**  | Cross symbol on arm          | 🏥    |
| **NEED**      | Grasping motion              | ✊    |
| **WANT**      | Reaching gesture             | 🫴    |
| **GO**        | Pointing away                | 👉    |
| **COME**      | Beckoning motion             | 🫳    |
| **MORE**      | Pinching together            | 🤌    |
| **STOP**      | Flat palm forward            | ✋    |

---

## 🌐 Supported Languages

| Language       | Code     | Example ("HELLO")     |
| -------------- | -------- | -------------------- |
| 🇺🇸 English    | `en-US`  | Hello                |
| 🇮🇳 Hindi      | `hi-IN`  | नमस्ते                |
| 🇮🇳 Marathi    | `mr-IN`  | नमस्कार               |
| 🇮🇳 Tamil      | `ta-IN`  | வணக்கம்               |
| 🇮🇳 Telugu     | `te-IN`  | నమస్కారం              |
| 🇮🇳 Kannada    | `kn-IN`  | ನಮಸ್ಕಾರ               |
| 🇮🇳 Bengali    | `bn-IN`  | নমস্কার               |

---

## 🔥 Troubleshooting

### ❌ Problem: `npm install` fails

```bash
# Clear npm cache and retry
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

### ❌ Problem: `pip install` fails for TensorFlow

```bash
# Upgrade pip first
pip install --upgrade pip

# If on Apple Silicon (M1/M2/M3), use:
pip install tensorflow-macos
```

### ❌ Problem: Camera not working

- Make sure no other app is using the camera (Zoom, FaceTime, etc.)
- Use **Google Chrome** for best compatibility
- Check browser permissions: `Settings → Privacy → Camera → Allow`

### ❌ Problem: Backend connection refused

- Verify the backend is running on **port 8000**
- The frontend expects the API at `http://localhost:8000`
- Check terminal for error messages

### ❌ Problem: `ModuleNotFoundError` in Python

```bash
# Make sure your virtualenv is activated
source venv/bin/activate    # You should see (venv) in your terminal prompt

# Then reinstall
pip install -r requirements.txt
```

### ❌ Problem: MediaPipe scripts fail to load

- Check your internet connection (MediaPipe loads from CDN)
- Try disabling ad blockers or browser extensions
- Check browser console (`F12` → Console tab) for errors

### ❌ Problem: Port already in use

```bash
# Kill process on port 8000 (backend)
lsof -ti:8000 | xargs kill -9

# Kill process on port 3000 (frontend)
lsof -ti:3000 | xargs kill -9
```

---

## 🌍 Deployment

### Frontend → Vercel

1. Push code to GitHub
2. Go to [vercel.com](https://vercel.com/) → **New Project**
3. Select your GitHub repository
4. Set **Framework Preset** → `Next.js`
5. Set **Root Directory** → `isl-translator/frontend`
6. Click **Deploy**

### Backend → Render

1. Go to [render.com](https://render.com/) → **New Web Service**
2. Link your GitHub repository
3. Set **Root Directory** → `isl-translator/backend`
4. Set **Build Command** → `pip install -r requirements.txt`
5. Set **Start Command** → `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. ⚠️ Update `allow_origins` in `main.py` with your Vercel URL

---

## 🛠️ Tech Stack

| Layer          | Technology                                      |
| -------------- | ----------------------------------------------- |
| **Frontend**   | Next.js 16, React 19, TypeScript, Tailwind CSS  |
| **Backend**    | Python, FastAPI, Uvicorn                         |
| **AI/ML**      | TensorFlow/Keras, MediaPipe Hands, Scikit-learn  |
| **Speech**     | Web Speech API (browser-native TTS)              |
| **Hand Detection** | MediaPipe Hands (21 landmarks, 63 coordinates) |

---

## 📜 API Endpoints

| Method | Endpoint    | Description                          |
| ------ | ----------- | ------------------------------------ |
| `GET`  | `/`         | Health check — confirms API is alive |
| `POST` | `/predict`  | Send 30-frame landmark sequence, get predicted gesture |
| `POST` | `/users`    | Create a new user profile            |
| `POST` | `/logs`     | Log a gesture detection event        |
| `GET`  | `/stats`    | Get platform statistics              |

**Example `/predict` request:**

```json
{
  "sequence": [
    [0.5, 0.3, 0.0, 0.6, 0.4, 0.1, ...],   // Frame 1: 63 values
    [0.5, 0.3, 0.0, 0.6, 0.4, 0.1, ...],   // Frame 2: 63 values
    // ... 30 frames total
  ]
}
```

**Response:**

```json
{
  "gesture": "HELLO",
  "confidence": 0.95
}
```

---

<div align="center">

**Built with ❤️ for accessibility**

Next.js · FastAPI · MediaPipe · TensorFlow · Web Speech API

</div>
