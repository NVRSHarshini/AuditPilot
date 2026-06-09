# AuditPilot

An autonomous compliance audit agent that compares corporate security policy documents against audit checklists, identifies gaps, scores risks, and generates a structured audit report — with a human-in-the-loop approval step before finalizing findings.

**Demo:** [YouTube walkthrough](https://www.youtube.com/watch?v=EEYMa4hJi2Q)

---

## Overview

AuditPilot accepts two documents — a corporate security policy and a compliance checklist (e.g. ISO 27001) — and runs them through a pipeline of ten specialized agents. Each agent handles a discrete step: ingesting and chunking the documents, extracting controls and requirements, embedding and storing them in MongoDB Atlas, retrieving semantically similar matches via cosine similarity, verifying compliance status, scoring risks, routing findings to responsible owners, pausing for human review, and finally generating a markdown audit report.

The system is built on LangGraph for agent orchestration, Gemini 1.5 Pro for language tasks, and MongoDB Atlas for vector storage and retrieval. All agents include local fallbacks so the pipeline continues functioning when API quotas are reached.

---

## Architecture

```
Document Upload (PDF)
        |
        v
[1] Document Ingestion Agent      -- Extracts text, splits into overlapping chunks
        |
        v
[2] Control Extraction Agent      -- Parses policy controls from policy chunks (Gemini)
        |
[3] Checklist Extraction Agent    -- Parses requirements from checklist chunks (Gemini)
        |
        v
[4] Embedding & Storage Agent     -- Generates 768-dim embeddings, stores in MongoDB Atlas
        |
        v
[5] Vector Retrieval Agent        -- Cosine similarity search: matches each requirement
        |                            to the closest policy control
        v
[6] Compliance Verification Agent -- Classifies each match as compliant /
        |                            partially_compliant / non_compliant (Gemini)
        v
[7] Risk Scoring Agent            -- Assigns Critical / High / Medium / Low to each gap
        |
[8] Owner Routing Agent           -- Routes each gap to responsible team + remediation advice
        |
        v
[9] Human-in-the-Loop Node        -- Pauses graph; reviewer approves, escalates, or accepts risk
        |
        v
[10] Report Generation Agent      -- Produces a full ISO 27001-style markdown audit report
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | TypeScript |
| Agent Orchestration | LangGraph (`@langchain/langgraph`) |
| LLM | Gemini 1.5 Pro via `@google/genai` |
| Vector Store | MongoDB Atlas (`compliance_controls` collection) |
| Embeddings | Gemini text-generation (768-dim) with local TF-IDF fallback |
| Server | Express.js |
| Frontend | Vite + React |
| PDF Parsing | `pdf-parse` |

---

## Prerequisites

- Node.js (v18 or later)
- A [Gemini API key](https://aistudio.google.com/app/apikey)
- A [MongoDB Atlas](https://www.mongodb.com/atlas) cluster with a database named `auditpilot`

---

## Setup

1. Clone the repository:

```bash
git clone https://github.com/NVRSHarshini/AuditPilot.git
cd AuditPilot
```

2. Install dependencies:

```bash
npm install
```

3. Copy the environment template and fill in your credentials:

```bash
cp .env.example .env.local
```

```env
GEMINI_API_KEY=your_gemini_api_key_here
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
```

4. Start the development server:

```bash
npm run dev
```

The app will be available at `http://localhost:3000`.

---

## Usage

1. Upload a corporate security policy document (PDF).
2. Upload a compliance checklist document (PDF) — for example, an ISO 27001 control set.
3. The agent pipeline runs automatically and streams progress.
4. When the pipeline reaches the human-in-the-loop step, review each finding and choose to **escalate** or **accept risk**.
5. Once all decisions are submitted, the report generation agent produces a full audit report in markdown.

---

## Agent Fallback Behavior

Each agent that calls the Gemini API has a deterministic local fallback. If the API returns a rate-limit error (429) or becomes unavailable, the agent falls back to heuristic text parsing or keyword-based logic. This ensures the pipeline completes even under quota constraints.

Embedding generation similarly falls back to a local term-frequency hash vectorizer when the Gemini embedding endpoint is unavailable.

The Gemini API call queue enforces a minimum 3-second gap between requests and applies exponential backoff with jitter on transient errors.

---

## Project Structure

```
AuditPilot/
├── server.ts           # All ten agents, LangGraph state machine, Express API routes
├── src/
│   └── types.ts        # Shared TypeScript interfaces
├── index.html          # App entry point
├── vite.config.ts      # Vite configuration
├── tsconfig.json
├── package.json
└── .env.example
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | API key for Gemini 1.5 Pro |
| `MONGODB_URI` | Yes | MongoDB Atlas connection string |

---

## Compliance Standards

The checklist extraction and report generation agents are designed with ISO 27001 control categories in mind, including Access Control, Data Encryption, Incident Response, Backup and Recovery, and Vendor Risk Management. The pipeline is not limited to ISO 27001 — any structured checklist document can be used as input.

---

## License

Apache 2.0 — see [LICENSE](./LICENSE) for details.
