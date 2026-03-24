# TopoMedic AI: Mitigating Shortcut Learning in Glaucoma Detection (IDSC 2026 Submission)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

**TopoMedic AI** is an interpretable diagnostic framework for Glaucomatous Optic Neuropathy (GON). It abandons the architectural bloat and "black-box" opacity of standard Deep Learning by pairing **Topological Data Analysis (TDA)** with an interpretable shallow classifier.

### The "Saliency Illusion"
In our adversarial audits, a transfer-learned ResNet-18 achieved a **0.99 AUROC** on the Hillel Yaffe Glaucoma Dataset (HYGD). However, targeted spatial ablation revealed this to be a fraud: the CNN maintained 96% accuracy even when the biological optic disc was completely masked. It was exploiting non-biological camera artifacts (Shortcut Learning). 

TopoMedic AI solves this. By anchoring predictions strictly to the mathematical invariants of retinal geometry, we built a model that is clinically safe, mathematically rigorous, and deployable on low-power edge hardware.

---

## ⚙️ The "Giant-Killer" Architecture

Rather than guessing which pixels matter, TopoMedic AI uses the physics of light and topology to force deterministic biomarker extraction.

1. **Automated ROI Localization:** A custom-tuned YOLO26 isolates the optic disc region, immediately stripping peripheral dataset noise.
2. **Orthogonal Landscape Engineering:** We extract three mathematically independent biological planes:
   - *Vascular:* Frangi vesselness filter (Green channel) for kinking/tortuosity.
   - *Disc Morphology:* High-sigma Gaussian blur (Red channel) for cup-to-disc boundary.
   - *Peripapillary Texture (PPA):* Shannon Entropy (grayscale) mapping spatial tissue degradation.
3. **Topological Data Analysis (Cubical Homology):** Using `giotto-tda`, we extract the birth/death of structural features across the landscapes. Aggressive downsampling to 5x5 **Persistence Images** acts as a topological low-pass filter, destroying micro-level visual noise while preserving macro-pathology.
4. **Interpretable Inference:** A Random Forest classifier processes these tabular topological vectors alongside scalar metadata (Image Quality Scores). 

---

## 📊 Technical & Clinical Benchmarks

| Metric / Benchmark | CNN Baseline (ResNet-18) | **TopoMedic AI (TDA+RF)** | Clinical Significance |
| :--- | :--- | :--- | :--- |
| **AUROC** | 0.99 | **0.873** | General Class Separability |
| **Recall (Sensitivity)**| 0.99 | **0.891** | **Patient Safety:** Avoids catastrophic False Negatives |
| **F2-Score** | 0.94 | **0.865** | **Clinical Utility:** Weights Recall 2x over Precision |
| **Ablation Resilience**| **FAILED** (Relies on noise) | **PASSED** (Pathology-bound) | Proves model logic is biologically valid |
| **Logic Mechanism** | Saliency only (Grad-CAM) | **Additive (SHAP)** | Generates "Itemized Clinical Receipts" |
| **Compute Profile** | High-TDP GPU Dependent | **O(n) CPU / Edge-Ready** | Air-gapped privacy; runs on a Raspberry Pi |

---

## 🖥️ Streamlit Deployment (Human-in-the-Loop)

TopoMedic AI is not just a script; it is a clinical tool. We built a Streamlit interface that provides **Itemised Transparency**. It outputs not just a probability score, but a game-theoretic **SHAP Waterfall Plot** binned by biological landscape (Vascular, Disc, Texture). This gives the clinician the exact biological rationale they need to authorize treatment.

### To Run App

uv run streamlit run app.py