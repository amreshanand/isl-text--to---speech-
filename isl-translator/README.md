# Indian Sign Language (ISL) Translator Platform

A real-world accessibility platform that translates Indian Sign Language (ISL) hand gestures into text and speech in real-time. Built specifically to help deaf and speech-impaired individuals communicate easily in classrooms, hospitals, workplaces, and public environments without requiring a human interpreter.

## Final System Architecture

```
Camera
  ↓
Frontend (Next.js)
  ↓
MediaPipe Hand Detection
  ↓
Gesture Recognition Model
  ↓
Text Output
  ↓
Translation (Multilingual)
  ↓
Text-to-Speech
  ↓
Audio Output
```

## Folder Structure

```
isl-translator/
├── frontend/             # Next.js web application
│   ├── app/              # App router (page.tsx, layout.tsx, etc.)
│   ├── public/           # Static assets
│   ├── package.json      # Frontend dependencies
│   └── ...
├── backend/              # FastAPI Python server
│   ├── main.py           # API endpoints
│   ├── requirements.txt  # Python requirements
│   └── ...
├── ai-model/             # AI model training and evaluation
│   ├── train.py          # Script for building the Keras model
│   └── saved_model/      # Output directory for the trained .h5 model
└── dataset/              # Dataset directory for video/landmarks
```

## Installation Instructions

### Prerequisites
* **Node.js**: v18 or later
* **Python**: v3.9 or later

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create an isolated Python environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### AI Model Setup

If you want to train your own custom ISL detection model:

1. Navigate to the ai-model directory:
   ```bash
   cd ai-model
   ```
2. Verify you have your Python environment activated.
3. Place initial sample datasets in `dataset/`.
4. Install equivalent backend dependencies (TensorFlow, OpenCV, MediaPipe, Scikit-learn).
5. Run the training script:
   ```bash
   python train.py
   ```

## Commands to Run the Project

### Start Backend API

Start the FastAPI application. It will load the AI model (or provide mock predictions for prototype) and listen for connections from the frontend.

```bash
cd backend
# With virtualenv activated
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
*API will run at http://localhost:8000*

### Start Frontend Client

Start the Next.js development server.

```bash
cd frontend
npm run dev
```
*Web App will run at http://localhost:3000*

Open `http://localhost:3000` in your web browser. Grant camera access to let the application start detecting gestures.

## Phase Explanations and Data Flow

1. **Phase 1: Simple Website with Camera**
   * **Mechanism**: Utilizes `navigator.mediaDevices.getUserMedia` in a React hook to stream video to a `<video>` element. 
   * **Connection**: Acts as the foundational data source for Phase 2.

2. **Phase 2: Hand Detection**
   * **Mechanism**: Binds `@mediapipe/camera_utils` to the active video source, piping frames continuously to `@mediapipe/hands`. Displays the original camera feed along with overlay drawings via `@mediapipe/drawing_utils` on an HTML5 `<canvas>`.
   * **Connection**: Translates raw video arrays into precise 3D (x, y, z) spacial coordinates for up to 21 hand landmarks. These 63 unique coordinates are exactly what the deep learning model requires.

3. **Phase 3: Gesture Recognition Model**
   * **Mechanism**: A Python script (`train.py`) aggregates frame-by-frame landmarks of specific gestures, standardizing them into an input array, and trains a multi-layer deep learning model using TensorFlow/Keras.
   * **Connection**: The resulting `.h5` model contains learned weights on how 63 coordinates map to specific sign language labels ("HELLO", "WATER").

4. **Phase 4: Real-Time Prediction**
   * **Mechanism**: Two systems communicating asynchronously. The React frontend bundles the 63 landmarks extracted per frame into JSON and sends a `POST /predict` to the FastAPI backend. FastAPI formats it, feeds it to the `model.predict()`, and returns the highest confident string.
   * **Connection**: Links the raw coordinate extraction (Frontend) with the trained intelligence (Backend).

5. **Phase 5 & 6: Text-to-Speech and Multilingual Translation**
   * **Mechanism**: Upon fetching a new gesture from Phase 4, the system translates the recognized token ("WATER" -> "पानी") using basic dictionary logic (which translates to Google Cloud APIs in production). It then constructs a `SpeechSynthesisUtterance` containing the localized string and triggers browser audio playback.
   * **Connection**: This transforms text output into an accessible, real-world utility format.

6. **Phase 7 & 8: Accessible UI & Database Storage**
   * **Mechanism**: Next.js state seamlessly updates the DOM with large, high-contrast, clear information. As an enhancement path, the backend specifies user tracking endpoints.
   * **Connection**: Transforms the engine into a holistic and scalable healthcare/educational platform ready for user adoption.

## Deployment Guide (Phase 9)

**Frontend (Vercel):**
1. Push your code to GitHub.
2. Sign up on [Vercel](https://vercel.com/) and create a "New Project".
3. Select your GitHub repository.
4. Set the "Framework Preset" to Next.js.
5. Set the "Root Directory" to `frontend`.
6. Deploy. Vercel will automatically run `npm run build`.

**Backend (Render):**
1. Sign up on [Render](https://render.com/).
2. Create a "New Web Service" and link your GitHub repository.
3. Set the "Root Directory" to `backend`.
4. Set the "Build Command" to `pip install -r requirements.txt`.
5. Set the "Start Command" to `uvicorn main:app --host 0.0.0.0 --port $PORT`.
6. Make sure to update the CORS settings in `main.py` to match the generated Vercel frontend URL.

**Database (Supabase):**
1. Sign up on [Supabase.com](https://supabase.com/).
2. Create a specific project database mapping to the schema referenced in `main.py`.
3. Add the resulting Postgres connection string directly into Rentder as a sensitive environment variable (e.g. `DATABASE_URL`). Implement SQLAlchemy or Asyncpg in `main.py` pointing to that URL.

**Testing in Production:**
Open the Vercel deployed URL on multiple devices (try a smartphone against a laptop) adjusting the lighting conditions to ensure the MediaPipe JavaScript bundle acts efficiently across mobile browsers. Check Render logs to ensure `POST /predict` API calls maintain a response latency under 200ms.
