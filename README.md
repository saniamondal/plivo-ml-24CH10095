# End-of-Turn (EoT) Detection System

This repository contains a high-performance, cross-lingual machine learning pipeline for detecting the end of a conversational turn in spoken audio. 

The model is designed for live conversational AI agents where low latency and high accuracy are critical. It successfully distinguishes between when a user has finished speaking (True EoT) and when they are merely pausing to think (Hold), preventing the AI agent from rudely interrupting.

## Performance Highlights
* **English Data:** 130 ms mean response delay (92% reduction vs baseline)
* **Hindi Data:** 100 ms mean response delay (94% reduction vs baseline)
* **Generalization:** ~800 ms response delay on unseen, out-of-domain Hindi audio.
* **Competition Metric:** Minimizes mean response delay while keeping false cut-offs (interrupting the user) under 5%.

## Architecture & Constraints
* **Fully Causal:** The feature extraction strictly uses only audio *before* the pause starts, preventing any future-data leakage.
* **Cross-Lingual Robustness:** Features use normalized relative trajectories (e.g., F0 slope ratios, energy decay slopes) rather than absolute values, ensuring strong transferability across languages with zero language-specific tuning.
* **Lightweight:** Runs purely on CPU using `numpy`, `scipy`, `librosa`, and `scikit-learn`—no heavy neural network inference required.
* **Model:** Gradient Boosting Classifier with Isotonic Calibration.

## Directory Structure
* `/eot_data`: Contains the raw audio and labels for both English and Hindi datasets.
* `/starter`: Contains the entire training pipeline, feature extraction logic, final serialized model, and rich summary reports. 
  * See `starter/SUMMARY.html` for a detailed breakdown of the features, metrics, and methodology.
  * See `starter/RUNLOG.md` and `starter/NOTES.md` for the experimental log.
