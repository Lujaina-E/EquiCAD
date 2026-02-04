# EquiCAD

## Overview

**EquiCAD** is a research-oriented system for detecting **sex-based bias against women** in Coronary Artery Disease (CAD) literature.  
It is designed to support fine-grained analysis of **text, tables, and figures** in biomedical PDFs, enabling systematic identification of biased clinical reasoning, diagnostic uncertainty, and symptom misinterpretation affecting women.

EquiCAD is intended for use in:
- Bias auditing of biomedical research papers
- Model evaluation and benchmarking on sex-based bias detection
- Exploratory analysis of diagnostic and reporting disparities in CAD research

EquiCAD requires the following versions to be installed to run (found in requirements.txt and runtime.txt, respectively):


  ```
  flask==3.0.0
  flask-cors==4.0.0
  openai==1.54.3
  python-dotenv==1.0.0
  PyPDF2==3.0.1
  gunicorn==21.2.0
  Pillow==10.1.0
  httpx==0.27.0
  PyMuPDF==1.26.7
  ```

```
python-3.11.8
```
  

### model/

#### `data/`
Contains raw and processed datasets used for training, testing, and evaluation.

- CSV files are considered **raw sources**
- JSONL files are **model-ready** and schema-validated

#### `scripts/`
Standalone Jupyter or Python scripts for:
- Converting CSV → JSONL
- Model evaluation

#### `config/`
Experiment and training configuration files.

- `settings.yaml` defines:
  - Model parameters
  - Training hyperparameters
  - Dataset paths
  - Checkpoints and evaluation settings

---

## Model task

The model classifies biomedical content into:

### Labels
- Bias
- No Bias

### Categories

**If Bias:**
- Sampling Bias
- Diagnostic Uncertainty / Bias
- Symptom Misinterpretation

**If No Bias:**
- Biological / Physiological Differences
- Factual / Neutral Observed Outcome

---

## Output format (strict)

All model outputs **must** follow this exact format:

Label: Bias | No Bias
Category: <exact category name>

_______________
# Local Deployment Using Docker
EquiCAD can be run locally using Docker or a manual development environment.

## Docker Deployment
#### Prerequisites

- **Docker**  
  https://www.docker.com/get-started

- **Docker Compose**  
  Bundled with Docker Desktop

## Docker Compose (bundled with Docker Desktop)

1. Create Environment Files in the Frontend Directory <br>
<br>
a) create a .env file in the directory /app/frontend/ and place this line:

`VITE_API_URL=http://0.0.0.0:5000/api`

  b) create a .env.production file in the directory /app/frontend/ and place this line: 

`VITE_API_URL=https://equicad-lexy.onrender.com/api`

2. Backend Environment Variables 
   
a) create a .env file in the directory /app/backend/ and place these variables one by one, ensuring no additional spaces. Remember to replace the OPEN_AI_KEY variable 

```
OPENAI_API_KEY=paste_key_here
FINE_TUNED_MODEL_ID=ft:gpt-4o-mini-2024-07-18:seall:equicad:D23JMHeE
REDIS_URL=redis://redis:6379/0
REDIS_HOST=redis
REDIS_PORT=6379
OCR_MODEL_ID=gpt-4o
```

3. Build and Start Containers
Run the following commands from inside the `app` directory: <br>
```
docker-compose down -v
docker-compose up --build
```

Access the app <br>
**Frontend:** http://localhost:3000
**Backend API:** http://localhost:5000/api

## Local Development (No Docker)

1. Navigate to the frontend directory my using the command `cd ~/EquiCAD/app/frontend`
2. In your terminal, run the command `npm install` and then `npm run dev`

3. Install the relevant dependencies by running the requirements.txt file though `cd ~/EquiCAD/app/backend` followed by `pip install -r requirements.txt`

4. make sure your environment variables { see 2.a) } have been setup
  
5. Run the backend using these commands in the terminal: <br>
- cd ~/EquiCAD/app/backend
- python app.py

_______________
## Access the Rendered App

The app has additionally been deployed using Render and can be accessed here: https://equicad-frontend.onrender.com/
Please be patient for the app to spin up on the servers, this should take around 1 minute of loading. During this time, a revolving "processing" symbol will be written on the page.  

Hosted Deployment
Access the deployed application: https://equicad-frontend.onrender.com/
