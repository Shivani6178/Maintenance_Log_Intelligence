# Aircraft Maintenance Log Intelligence System

An end-to-end intelligent maintenance system that combines **predictive RUL forecasting** with **RAG-based historical incident retrieval** to help aircraft maintenance engineers make faster, data-driven decisions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Future Improvements](#future-improvements)
- [Authors](#authors)

---

## Overview

This system addresses two critical problems in aircraft maintenance:

### Problem 1: Predictive Maintenance
**Challenge**: Engines fail unexpectedly, leading to unscheduled downtime and safety risks.

**Solution**: A Bi-LSTM deep learning model predicts remaining useful life (RUL) from 30 consecutive cycles of 14 engine sensor readings, giving maintenance teams advance warning to schedule repairs on their terms.

### Problem 2: Institutional Knowledge Discovery
**Challenge**: Accumulated lessons from thousands of real incidents are trapped in PDFs and institutional memory, inaccessible in real-time.

**Solution**: A hybrid RAG (Retrieval Augmented Generation) system searches 9,400+ real ASRS maintenance incident reports, retrieves the most relevant ones using semantic search + keyword matching + cross-encoder reranking, and synthesizes them into a grounded, cited answer using an LLM.

---

## Features

### RUL Prediction (`/predict-rul`)
- **Input**: CSV of sensor data (30 timesteps × 14 sensors, most recent cycle last)
- **Output**: Predicted remaining cycles, cycles remaining, and warning level (NORMAL / WARNING / CRITICAL)
- **Model**: Bidirectional LSTM trained on NASA CMAPSS turbofan degradation data
- **Performance**: 15.82 RMSE (10.9% improvement over XGBoost baseline)
- **Safety threshold**: Warnings triggered at RUL < 50 cycles (warning) or < 20 cycles (critical)

### Maintenance Log Retrieval (`/query`)
- **Input**: Natural language question about engine/maintenance issues
- **Output**: Grounded answer with 5 cited ASRS incident reports (ACN numbers, aircraft type, flight phase, relevance scores)
- **Architecture**: Vector search (BGE-M3) + keyword search (Postgres full-text) → RRF fusion → cross-encoder reranking → Groq LLM generation
- **Grounding**: System correctly declines out-of-scope questions rather than hallucinating

### Interactive UI
- **Streamlit frontend** with two tabs: RUL prediction and maintenance log search
- **Error handling**: Clear, actionable messages for malformed input or API errors
- **Export-ready**: Upload CSVs, download results

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface (Streamlit)                  │
│  ┌──────────────────────┐          ┌──────────────────────────┐    │
│  │  RUL Prediction Tab  │          │ Maintenance Search Tab    │    │
│  │  (Upload CSV)        │          │ (Natural language query)  │    │
│  └──────────────┬───────┘          └──────────────┬───────────┘    │
└─────────────────┼──────────────────────────────────┼────────────────┘
                  │                                  │
        ┌─────────▼──────────┐         ┌────────────▼─────────┐
        │  FastAPI Backend   │         │  FastAPI Backend     │
        │  /predict-rul      │         │  /query              │
        └────────┬───────────┘         └──────────┬───────────┘
                 │                                │
        ┌────────▼──────────────────────────────▼────┐
        │      Core Logic Layer (src/)              │
        │  ┌────────────────────────────────────┐   │
        │  │ RUL: BiLSTM + MinMaxScaler         │   │
        │  └────────────────────────────────────┘   │
        │  ┌────────────────────────────────────┐   │
        │  │ RAG:                               │   │
        │  │  • vector_search (BGE-M3)          │   │
        │  │  • keyword_search (Postgres FTS)   │   │
        │  │  • fusion (RRF)                    │   │
        │  │  • reranker (cross-encoder)        │   │
        │  │  • generation (Groq LLM)           │   │
        │  └────────────────────────────────────┘   │
        └────────┬──────────────────────────┬───────┘
                 │                          │
        ┌────────▼──────────┐    ┌──────────▼──────┐
        │  PostgreSQL       │    │  Model Files    │
        │  + pgvector       │    │  (/models)      │
        │  ┌──────────────┐ │    │ ┌────────────┐  │
        │  │ chunks table │ │    │ │ bilstm.h5  │  │
        │  │ HNSW index   │ │    │ │ scaler.pkl │  │
        │  │ GIN FTS idx  │ │    │ └────────────┘  │
        │  └──────────────┘ │    │                 │
        └───────────────────┘    └─────────────────┘
```

### Component Responsibilities

- **`src/retrieval/`**: Retrieval logic (vector search, keyword search, fusion, reranking)
- **`src/generation/`**: LLM-based answer generation and grounding
- **`src/api/`**: FastAPI routes, schema definitions, resource management (lifespan)
- **`app/`**: Streamlit frontend, user interaction
- **`models/`**: Saved Bi-LSTM and MinMaxScaler artifacts
- **`docker/`**: Dockerfile and docker-compose for containerization

---

## Datasets

### 1. CMAPSS (NASA Turbofan Degradation Data)

**Source**: [NASA's Commercial Modular Aero-Propulsion System Simulation](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

**What**: Simulated turbofan engine sensor data from controlled run-to-failure experiments.

**Why chosen**:
- Industry-standard benchmark for RUL prediction research
- Clean, labeled data with ground-truth failure times
- Directly credible for portfolio ("trained on NASA-published benchmark")

**Our subset (FD001)**:
- 100 training engines, 100 test engines
- 1 operating condition (simplest variant)
- 21 sensors per timestep
- ~9,400 training sequences after preprocessing

**Preprocessing**:
- Dropped 7 constant sensors (zero variance): s_1, s_5, s_6, s_10, s_16, s_18, s_19
- Retained 14 sensors with meaningful degradation signals
- Labeled RUL as piecewise linear with ceiling at 125 cycles (physics-informed, standard in literature)
- Scaled with MinMaxScaler (range [-1, 1]), fitted on training data only

### 2. ASRS (Aviation Safety Reporting System)

**Source**: [NASA's ASRS Database](https://asrs.arc.nasa.gov)

**What**: 2M+ voluntary, confidential incident reports from pilots, mechanics, ATC since 1976.

**Why chosen**:
- Authentic maintenance language and real incident patterns
- Publicly available — no proprietary data concerns
- Direct relevance to aircraft maintenance engineering
- Large scale enables meaningful retrieval

**Our subset**:
- Filtered to 9,443 engine-maintenance-focused reports (2009–2026)
- Keywords: "engine AND (vibration OR stall OR EGT OR bearing)" + "maintenance history" + "recurring issues"
- Excluded pre-2009 (outdated aircraft/procedures)

**Processing**:
- De-duplicated by ACN (Accession Number)
- Cleaned empty narratives
- Split into 2 chunk types per report:
  - **Narrative**: full incident description (~350 words, 14 sensors raw)
  - **Synopsis**: analyst-written summary (~220 words)
- Result: 18,886 retrievable chunks (9,443 narrative + 9,443 synopsis)

---

## Models

### RUL Prediction: Bi-LSTM

**Why Bi-LSTM?**
- Turbofans degrade over time; **sequences matter**
- Bidirectional LSTM captures temporal patterns in both forward and backward directions
- Sequence-to-scalar architecture (input: 30 timesteps, output: single RUL value)
- More parameter-efficient than Transformers on small datasets (100 engines, ~9.4K sequences)

**Architecture**:
```
Input (batch, 30 cycles, 14 sensors)
    ↓
BiLSTM(128, return_sequences=True) 
    ↓
BatchNormalization
    ↓
Dropout(0.3)
    ↓
BiLSTM(64, return_sequences=False)
    ↓
BatchNormalization
    ↓
Dropout(0.3)
    ↓
Dense(32, relu)
    ↓
Dropout(0.2)
    ↓
Dense(1, relu) → Output: RUL ≥ 0
```

**Training**:
- Loss: Huber(delta=10.0) — robust to outlier engines
- Optimizer: Adam(lr=0.001)
- Callbacks:
  - EarlyStopping(patience=10, restore_best_weights=True)
  - ReduceLROnPlateau(factor=0.5, patience=5)
- Split: engine-level (not row-level) to prevent sequence leakage
- Train/Val: 85/15 split, seed=42

**Performance**:
| Baseline | RMSE | MAE | NASA Score |
|---|---|---|---|
| Random Forest | 18.21 | 13.30 | 922.32 |
| XGBoost | 17.75 | 12.96 | 840.73 |
| **Bi-LSTM** | **15.82** | **11.50** | **527.73** |

- **10.9% RMSE improvement** vs XGBoost
- **NASA asymmetric score**: penalizes overestimation (exp(d/10)−1) and underestimation (exp(−d/13)−1) differently; safety-critical to avoid predicting too much life remaining
- Median error in critical range (RUL 0–25): 2.12 cycles

**Design decision**: Tested Bi-LSTM + Bahdanau attention, but Wilcoxon rank-sum showed p=0.57 (not statistically significant). Worse in critical range (error 2.12 → 4.35). **Rejected added complexity without evidence** — a key lesson in production ML.

### RAG Retrieval: Hybrid Search + Reranking

**Pipeline**:

1. **Vector Search (BGE-M3)**
   - Model: `BAAI/bge-m3`, 1024-dimensional embeddings
   - Why: SOTA on MTEB leaderboard; supports dense + sparse retrieval; 8K token context
   - Index: pgvector HNSW (Hierarchical Navigable Small World) — O(log n) query latency
   - Top-k: 50 candidates

2. **Keyword Search (Postgres Full-Text)**
   - Method: `to_tsquery('english', query)` with OR logic across terms
   - Index: GIN (Generalized Inverted Index) on tsvector column
   - Why: Catches exact technical terms (EGT, MEL, N1) that semantic search blurs
   - Top-k: 50 candidates

3. **RRF Fusion**
   - Method: Reciprocal Rank Fusion — rank-based (not score-based)
   - Score: Σ(1/(k + rank)) across retrieval methods, k=60
   - Why: Avoids normalizing incompatible score ranges (cosine similarity vs ts_rank)
   - Result: Single ranked list of top-50

4. **Cross-Encoder Reranking**
   - Model: `BAAI/bge-reranker-v2-m3`
   - Why: Reads query + passage together (not independently); ~10–15% improvement in recall@5
   - Input: all 50 candidates
   - Output: top-5 with relevance scores (0–1, higher = more relevant)

5. **LLM Generation (Groq)**
   - Model: `llama-3.3-70b-versatile` (default)
   - System prompt: Forces per-claim ACN citations, refuses out-of-scope questions
   - Input: top-5 chunks + user query
   - Output: grounded answer with inline citations

**Example retrieval result**:
Query: "engine vibration and bearing wear during climb"

| Rank | ACN | Aircraft | Flight Phase | Rerank Score | Content |
|---|---|---|---|---|---|
| 1 | 1955210 | B737-800 | Cruise | 0.888 | "Pilot reported engine vibration during final climb..." |
| 2 | 1029635 | B757-200 | Climb | 0.864 | "...felt airframe vibration, #2 engine vibration jumped from .3 to 3.0..." |
| 3 | 2309187 | B767-300 | Climb | 0.821 | "In climb with climb power set excessive vibration left engine 5.0..." |
| 4 | 2120521 | EMB 170 | Climb | 0.819 | "...#2 engine experienced high vibration levels during climb..." |
| 5 | 926964 | B737-700 | Initial Climb | 0.799 | "...felt thump on airframe, vibration. Noted #1 engine vibration..." |

**Key observation**: All 5 results are directly relevant. System correctly notes "no specific information about bearing wear" — demonstrates grounding, not hallucination.

---

## Installation

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Git
- ~10 GB disk space (for model downloads)

### Local Setup (Development)

**1. Clone and navigate to project**:
```bash
git clone https://github.com/yourusername/Maintenance_Log_Intelligence.git
cd Maintenance_Log_Intelligence
```

**2. Create virtual environment**:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux
```

**3. Install dependencies**:
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your credentials:
# - GROQ_API_KEY from https://console.groq.com
# - POSTGRES_PASSWORD (local dev password)
# - DATABASE_URL (PostgreSQL connection string)
```

**5. Start PostgreSQL with pgvector**:
```bash
docker compose -f docker/docker-compose.yml up -d postgres
```

**6. Initialize database schema** (runs RAG_02):
```bash
# Open Jupyter in notebooks/ folder
jupyter notebook notebooks/
# Run RAG_02_Database_Setup.ipynb (creates chunks table + indexes)
```

**7. Populate database** (requires pre-embedded chunks from RAG_03):
```bash
# If you have asrs_embeddings.npy + asrs_chunks_metadata.csv from Colab:
python -m tests.test_retrieval  # ingests into local Postgres
```

**8. Start FastAPI backend**:
```bash
uvicorn src.api.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for Swagger UI.

**9. Start Streamlit frontend** (in a new terminal):
```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`.

### Docker Setup (Production)

**Build and run everything**:
```bash
docker compose -f docker/docker-compose.yml up --build
```

This spins up:
- PostgreSQL + pgvector (port 5432)
- FastAPI (port 8000)
- Streamlit (port 8501)

All services share environment variables from `.env`.

---

## Usage

### RUL Prediction

**Via Streamlit UI**:
1. Open `http://localhost:8501`
2. Go to "📉 RUL Prediction" tab
3. Upload a CSV: 30 rows × 14 sensor columns
4. Click "Predict RUL"
5. View predicted cycles, warning level (NORMAL / WARNING / CRITICAL)

**Via cURL**:
```bash
curl -X POST http://localhost:8000/predict-rul \
  -H "Content-Type: application/json" \
  -d '{"sensor_readings": [[s0, s1, ..., s13], [...], ...]}'  # 30 rows
```

**Expected response**:
```json
{
  "predicted_rul": 47.32,
  "cycles_remaining": 47,
  "warning_level": "normal"
}
```

### Maintenance Log Search

**Via Streamlit UI**:
1. Open `http://localhost:8501`
2. Go to "🔍 Maintenance Log Search" tab
3. Type your question, e.g., "What causes compressor stalls during climb?"
4. Click "Search"
5. View synthesized answer + 5 cited ASRS incident reports

**Via cURL**:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "engine vibration during climb", "top_k": 5}'
```

**Expected response**:
```json
{
  "query": "engine vibration during climb",
  "answer": "Engine vibration during climb was reported in multiple incidents...",
  "sources": [
    {
      "acn": "1955210",
      "aircraft_model": "B737-800",
      "flight_phase": "Cruise",
      "excerpt": "Pilot reported engine vibration during final climb...",
      "relevance_score": 0.888
    },
    ...
  ]
}
```

---

## API Documentation

### Endpoints

#### `GET /health`
Health check. Returns:
```json
{"status": "ok", "rul_model_loaded": true}
```

#### `POST /query`
Maintenance log search. 

**Request**:
```json
{
  "query": "engine vibration and bearing wear during climb",
  "top_k": 5
}
```

**Parameters**:
- `query` (str, required): Natural language question, min 3 chars
- `top_k` (int, optional): Number of sources to retrieve (1–20, default 5)

**Response**:
```json
{
  "query": "...",
  "answer": "...",
  "sources": [
    {"acn": "...", "aircraft_model": "...", "flight_phase": "...", "excerpt": "...", "relevance_score": 0.88},
    ...
  ]
}
```

#### `POST /predict-rul`
RUL prediction.

**Request**:
```json
{
  "sensor_readings": [[...], [...], ...]  // 30 rows × 14 columns
}
```

**Parameters**:
- `sensor_readings` (list[list[float]], required): 30 timesteps × 14 sensors

**Response**:
```json
{
  "predicted_rul": 47.32,
  "cycles_remaining": 47,
  "warning_level": "normal"
}
```

**Warning levels**:
- `normal`: RUL ≥ 50 cycles
- `warning`: 20 ≤ RUL < 50 cycles
- `critical`: RUL < 20 cycles

For full interactive documentation, visit `/docs` after starting the API.

---

## Results

### RUL Model Performance

- **Test RMSE**: 15.82 cycles (vs. 17.75 XGBoost baseline, **10.9% improvement**)
- **Test MAE**: 11.50 cycles
- **NASA Asymmetric Score**: 527.73 (competitive with published benchmarks)
- **Critical-range error** (RUL 0–25): median 2.12 cycles
- **Competitive position**: Published Bi-LSTM SOTA ≈ 17.6 RMSE; this model ≈ 15.82

### RAG System Performance

- **Retrieval coverage**: 18,886 chunks from 9,443 real ASRS reports
- **Query latency**: ~30s cold start (models loading), <2s warm start
- **Answer grounding**: System correctly declines out-of-scope questions; no hallucinations on test queries
- **Citation accuracy**: 100% of generated claims traceable to ACN numbers

### Example Query Results

**Query**: "Engine vibration and bearing wear during climb"

**Generated Answer**:
> "Engine vibration during climb was reported in multiple incidents, including excessive vibration levels (ACN 1955210, ACN 1029635, ACN 2309187, ACN 2120521, ACN 926964). In some cases, reducing power helped decrease or stop the vibration (ACN 1955210, ACN 2120521, ACN 926964). However, the context does not provide specific information on bearing wear during climb. The reports only mention vibration levels and the actions taken by flight crew to address the issue."

**Sources** (with relevance scores):
1. ACN 1955210 (B737-800, Cruise, 0.888)
2. ACN 1029635 (B757-200, Climb, 0.864)
3. ACN 2309187 (B767-300, Climb, 0.821)
4. ACN 2120521 (EMB 170, Climb, 0.819)
5. ACN 926964 (B737-700, Initial Climb, 0.799)

**Key observation**: System correctly flags the gap ("no specific information on bearing wear") rather than fabricating an answer — demonstrates safety-conscious grounding.

---

## Project Structure

```
Maintenance_Log_Intelligence/
├── notebooks/
│   ├── RAG_01_Data_Exploration.ipynb              # ASRS data cleaning & stats
│   ├── RAG_02_Database_Setup.ipynb                # pgvector schema creation
│   ├── RAG_03a_Embedding_Generation.ipynb         # Colab: embed chunks with BGE-M3
│   ├── RAG_03b_Database_Ingestion.ipynb           # Local: insert embeddings into pgvector
│   ├── CMAPSS_FD001_EDA_and_Baselines.ipynb       # RUL EDA + RF/XGBoost baselines
│   ├── CMAPSS_FD001_BiLSTM_Sequence_Model.ipynb   # RUL: Bi-LSTM training
│   └── CMAPSS_FD001_Attention_BiLSTM.ipynb        # RUL: attention variant (rejected)
│
├── src/
│   ├── __init__.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── db.py                                  # Postgres connection management
│   │   ├── vector_search.py                       # BGE-M3 dense retrieval
│   │   ├── bm25_search.py                         # Keyword search (Postgres full-text)
│   │   ├── fusion.py                              # RRF (Reciprocal Rank Fusion)
│   │   ├── reranker.py                            # Cross-encoder reranking
│   │   └── pipeline.py                            # Orchestration (retrieve + rerank)
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm_client.py                          # Groq API wrapper + grounding
│   │   └── rag_service.py                         # Full RAG loop (retrieve + generate)
│   └── api/
│       ├── __init__.py
│       ├── schemas.py                             # Pydantic request/response models
│       └── main.py                                # FastAPI app, routes, lifespan
│
├── app/
│   └── streamlit_app.py                           # Streamlit UI (RUL + search tabs)
│
├── docker/
│   ├── Dockerfile.api                             # FastAPI container
│   ├── Dockerfile.streamlit                       # Streamlit container
│   └── docker-compose.yml                         # Orchestrate all services
│
├── dataset/
│   ├── raw/
│   │   └── asrs_reports/                          # Raw ASRS CSVs (from ASRS export)
│   ├── processed/
│   │   ├── asrs_cleaned.csv                       # Cleaned 9,443 reports
│   │   ├── asrs_chunks_metadata.csv               # 18,886 chunks + metadata
│   │   └── asrs_embeddings.npy                    # 18,886 × 1024 BGE-M3 vectors
│   └── golden_qa/
│       └── qa_pairs.json                          # (Future) hand-labeled Q&A for RAGAS eval
│
├── models/
│   ├── bilstm_model.h5                            # Trained Bi-LSTM RUL model
│   ├── scaler.pkl                                 # MinMaxScaler for RUL input normalization
│   ├── rf_baseline.pkl                            # Random Forest baseline (reference)
│   └── xgb_baseline.pkl                           # XGBoost baseline (reference)
│
├── tests/
│   ├── __init__.py
│   ├── test_retrieval.py                          # End-to-end retrieval pipeline test
│   └── test_generation.py                         # End-to-end RAG generation test
│
├── .github/
│   └── workflows/
│       └── deploy.yml                             # GitHub Actions CI/CD (TODO)
│
├── .env.example                                    # Template for environment variables
├── .gitignore
├── requirements.txt                                # Python dependencies
├── README.md                                       # This file
└── docker-compose.yml                             # Local development orchestration
```

---

## Technologies

### Data & ML
- **TensorFlow/Keras**: Bi-LSTM RUL model training and inference
- **scikit-learn**: MinMaxScaler for data normalization
- **Hugging Face Transformers**: BGE-M3 (embedding), BGE-reranker-v2-m3 (cross-encoder)
- **NumPy, Pandas**: Data processing

### Database & Vector Search
- **PostgreSQL**: Relational database, runs locally or on AWS RDS
- **pgvector**: Vector extension for PostgreSQL (HNSW index for O(log n) retrieval)
- **SQLAlchemy**: ORM (optional, used for schema generation)

### Retrieval & Generation
- **LangChain**: RAG orchestration (optional; used in early exploration, now direct Python)
- **Groq**: LLM API for answer generation (free tier available)
- **Rank-BM25 / Postgres full-text**: Keyword search

### Backend API
- **FastAPI**: High-performance async web framework
- **Uvicorn**: ASGI server for FastAPI
- **Pydantic**: Request/response validation

### Frontend
- **Streamlit**: Interactive web UI (two-tab layout for RUL and search)
- **Requests**: HTTP client for calling FastAPI from Streamlit

### Infrastructure
- **Docker**: Containerization (API, Streamlit, PostgreSQL)
- **Docker Compose**: Orchestrate multi-container local development
- **GitHub Actions**: CI/CD pipeline (TODO)


---


## Authors

**Sanjana Dubey** — AI/ML Engineer
- GitHub: [Shivani6178](https://github.com/Shivani6178)
- LinkedIn: [https://www.linkedin.com/in/sanjana-dubey/](https://www.linkedin.com/in/sanjana-dubey-aa0358255/)
- Email: [dubeysanjana23@gmail.com]

---

## Acknowledgments

- **NASA**: CMAPSS turbofan degradation dataset
- **FAA/NASA**: ASRS incident database
- **BAAI**: BGE-M3 embedding model and BGE-reranker-v2-m3 cross-encoder
- **HuggingFace**: Transformers library and model hosting
- **Groq**: Groq for LLM inference (free tier)

---


## Quick Start

```bash
# Clone
git clone https://github.com/yourusername/Maintenance_Log_Intelligence.git
cd Maintenance_Log_Intelligence

# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# Run (all services)
docker compose -f docker/docker-compose.yml up --build

# Access
# - API Docs: http://localhost:8000/docs
# - Streamlit UI: http://localhost:8501
# - Postgres: localhost:5432
```

