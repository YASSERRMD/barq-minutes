# Privacy

barq-minutes is designed as a local-first browser application. The project has no backend, no serverless functions, and no telemetry code.

## Data Flow

1. The browser captures microphone audio through the Web Media APIs.
2. The ASR model is loaded in the browser and used for local transcription.
3. Transcript windows are sent only to the local LLM runtime in the same browser context.
4. Extracted decisions, action items, open questions, summary bullets, and transcript turns are saved in IndexedDB.
5. Search indexes are built locally with MiniLM embeddings and `barq-vweb`.
6. Markdown and PDF exports are generated locally in the browser.

## Network Access

The intended network use is limited to first-time model and package asset loading from the configured model origins and the static application host. No meeting audio, transcript text, extracted content, search query, or exported document is sent to an external inference API.

The application does not call OpenAI, Anthropic, Hugging Face Inference, or any other hosted model API.

## Audio Storage

Audio blobs are not stored by default. The recording screen includes an opt-in toggle named "Store audio blob". When disabled, the meeting record stores transcript and structured outputs only.

When enabled, audio blobs are stored in IndexedDB under an `audio:` key. The settings screen can clear stored meeting and audio records.

## IndexedDB Storage

Meeting records include:

- Meeting metadata
- Transcript turns
- Decisions
- Action items
- Open questions
- Summary bullets
- Tags
- Optional audio blob key

The app uses `idb-keyval` for structured meeting and audio records. Vector retrieval data is stored through `barq-vweb`.

## User Controls

- Audio storage is opt-in.
- Clear all data is available from settings.
- Markdown and PDF export files are generated only when the user clicks export.

## Deployment Requirement

SharedArrayBuffer and WASM threading require cross-origin isolation. Production hosting must serve:

- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Embedder-Policy: require-corp`

Without those headers, some browser-local model and WASM paths may fail or run slowly.
