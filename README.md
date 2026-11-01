# NABRA AI | نبرة

**AI-powered analysis of recorded customer-service calls.**

NABRA AI converts recorded conversations into transcripts, identifies speakers, classifies customer and agent roles, summarizes the conversation, and presents results in a Streamlit dashboard.

## Components

- **Whisper** — speech-to-text transcription.
- **Pyannote** — speaker diarization.
- **DistilBERT** — agent/customer role classification.
- **Qwen** — call summarization and resolution analysis.
- **Streamlit** — interactive call analysis, history, and visualizations.

## Evaluation (latest Colab run)

These are results from the latest tested sample, **not end-to-end project accuracy**:

- Whisper: word error rate (WER) **14.00%**, character error rate (CER) **9.98%** across seven test calls.
- Speaker-role attribution: **96.91% accuracy**, **96.74% macro-F1** on 711 evaluated segments across seven calls, with **98.20% reference coverage**.

The latest sample differs from the sample used in the original technical report; its scores must not be presented as the original report's results. Qwen resolution labels may need human review where evidence is absent.

## Running the project

Use the Colab notebook with a GPU runtime and install the documented dependencies. Pyannote requires an approved Hugging Face account and an `HF_TOKEN` supplied privately via Colab Secrets. The downloaded pretrained models and the fine-tuned DistilBERT weights must be available before running the complete pipeline.

The repository should contain **source code and documentation only**, not customer audio, raw transcripts, access tokens, or large model checkpoints. Project artifacts and models should be transferred through an authorized private channel separately.

> NABRA AI currently analyzes **recorded calls**; it is not a live voice assistant that can execute customer-account actions.
