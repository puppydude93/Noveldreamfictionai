# NovelDreamFiction AI — Vertex AI Chat

A Node.js chat interface powered by Google Vertex AI (Gemini), with streaming support and a structured system prompt.

## Prerequisites

- Node.js >= 18
- A Google Cloud project with the Vertex AI API enabled
- Authentication: `gcloud auth application-default login` or a service account key

## Setup

1. Install dependencies

   ```
   npm install
   ```

2. Configure environment variables (copy and edit the example)

   ```
   cp .env.example .env
   ```

   | Variable | Required | Default | Description |
   |---|---|---|---|
   | `GOOGLE_CLOUD_PROJECT` | Yes | — | Your GCP project ID |
   | `GOOGLE_CLOUD_LOCATION` | No | `us-central1` | Vertex AI region |
   | `VERTEX_MODEL` | No | `gemini-1.5-pro` | Gemini model name |

## Usage

Run with the default prompt:

```
npm start
```

Pass a custom prompt as arguments:

```
node app.js "Explain the three-act story structure"
```

## How it works

- Sends the prompt to Gemini via the Vertex AI SDK using streaming (`generateContentStream`).
- Falls back to non-streaming (`generateContent`) if streaming is unavailable.
- Accepts optional chat history as a second argument to `chatOnce()` for multi-turn use.
