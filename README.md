# 3D Heritage Reconstruction System

This project is a system for 3D reconstruction of heritage sites using photogrammetry and Gaussian Splatting. It consists of a Vue.js frontend and a set of backend services deployed on Modal.

## Prerequisites

- **Node.js** (v18+ recommended)
- **Python** (v3.10+ recommended)
- **Modal Account**: You need a [Modal](https://modal.com/) account to deploy the backend.
- **Modal CLI**: Installed and authenticated (`pip install modal`, `modal setup`).

## Backend Deployment

The backend is composed of multiple Modal applications that communicate with each other. You must deploy them in the following order to ensure dependencies are met.

1. **Deploy Model Inference Service**
   This service runs the Pi3 model for initial 3D point cloud generation.
   ```bash
   modal deploy backend/model_endpoint.py
   ```

2. **Deploy WebSocket & Worker Service**
   This service handles real-time updates and processes the reconstruction queue.
   ```bash
   modal deploy backend/websocket_light.py
   ```
   > **Note:** After deployment, Modal will display a URL ending in `.modal.run` (e.g., `https://your-username--websocket-light-fastapi-app.modal.run`). Copy this URL; you will need it for the frontend configuration as `VITE_WS_API_URL`.

3. **Deploy Splatting Service**
   This service handles training of Gaussian Splatting models.
   ```bash
   modal deploy backend/splat_server.py
   ```

4. **Deploy Baking Service**
   This service handles conversion to COLMAP and Bundle Adjustment.
   ```bash
   modal deploy backend/baking_server.py
   ```

5. **Deploy Main API Service**
   This is the main entry point for the frontend, handling scene management and file uploads.
   ```bash
   modal deploy backend/modal_server.py
   ```
   > **Note:** Copy the URL displayed after deployment (e.g., `https://your-username--ut3c-heritage-backend-brute-fastapi-app.modal.run`). You will need it for the frontend configuration as `VITE_API_BASE_URL`.

## Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Configure Environment Variables:**
   Create a file named `.env` in the `frontend/` directory (you can copy `.env.example` if it exists, or create new) and add the backend URLs you obtained during deployment:

   ```ini
   # URL from step 5 (Main API)
   VITE_API_BASE_URL=https://your-username--ut3c-heritage-backend-brute-fastapi-app.modal.run

   # URL from step 2 (WebSocket Service)
   VITE_WS_API_URL=https://your-username--websocket-light-fastapi-app.modal.run
   ```

4. **Run the Development Server:**
   ```bash
   npm run dev
   ```

5. **Access the App:**
   Open your browser and navigate to the URL shown in the terminal (usually `http://localhost:5173`).

## System Architecture

- **Frontend**: Vue.js 3 + Vite + Three.js (for PLY visualization).
- **Backend (Modal)**:
  - `ut3c-heritage-backend-brute`: Main REST API (FastAPI).
  - `websocket-light`: Manages WebSockets for real-time updates and orchestrates background workers.
  - `pi3-inference`: GPU-accelerated inference using Pi3.
  - `ut3c-heritage-baking`: Handles heavy processing tasks like COLMAP conversion and Bundle Adjustment.
  - `ut3c-heritage-splat`: Handles training of 3D Gaussian Splatting models.
- **Storage**: All services share a Modal Volume named `ut3c-heritage`.

## Usage

1. Open the web interface.
2. Click on "Crear Nueva Escena Compartida" to create a new project.
3. Upload images to the scene. The system will automatically:
   - Process the images using the Pi3 model.
   - Generate a point cloud.
   - Update the viewer in real-time via WebSockets.
4. For advanced reconstruction, use the available tools in the UI (if exposed) or API endpoints to trigger COLMAP conversion and Gaussian Splatting training.

