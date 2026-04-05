# Multilingual Code-Mixed Text Summarization (Indic Languages + English)

## 📌 Project Overview

This project aims to build a **deep learning-based abstractive text summarization system** capable of handling **code-mixed text across multiple Indian languages and English** (e.g., Hinglish, Tanglish, etc.).

The system will:

* Accept **code-mixed multilingual input**
* Understand mixed linguistic structure
* Generate **concise summaries**
* Work **without using external translation APIs**

---

## 🎯 Objectives

* Build a **transformer-based summarization model**
* Handle **code-mixed inputs (Hindi-English, Tamil-English, etc.)**
* Leverage **monolingual + code-mixed datasets**
* Use **synthetic data generation** to overcome data scarcity
* Deliver a **working, efficient system within 1 month**

---

## 🧠 Core Approach

We use a **hybrid training strategy**:

1. Learn summarization from **monolingual data (ILSUM)**
2. Introduce code-mixing using **synthetic data**
3. Improve robustness using **real code-mixed datasets**
4. Fine-tune a **multilingual transformer (IndicBART)**

---

## 🤖 Model Selection

### ✅ Model: IndicBART (`ai4bharat/IndicBART`)

**Why IndicBART?**

* Designed for Indian languages
* Supports multilingual generation
* Efficient (~244M parameters)
* Handles script diversity better than generic models

---

## 📊 Datasets Used

### 1. ILSUM Dataset (Primary)

* Monolingual Indian language articles + summaries
* Used to teach **core summarization ability**

### 2. Code-Mixed Dataset (Secondary)

Choose ONE:

* Hinglish dataset OR
* GupShup dataset

Used to:

* Expose model to **real code-mixed patterns**

---

## 🔧 Data Processing Pipeline

### Step 1: Basic Preprocessing

* Remove extra spaces
* Remove noisy/unwanted characters
* Normalize text (minimal)

### Step 2: Optional Transliteration

* Convert Romanized Indic words → native script
* Tool: `indic-transliteration`

Example:

```
"aaj weather acha hai"
→ "आज weather अच्छा है"
```

⚠️ Keep preprocessing simple to avoid complexity

---

## 🔥 Synthetic Code-Mixed Data Generation

### Purpose:

Create code-mixed training data from ILSUM

### Method:

* Select 20–40% words randomly
* Replace with equivalent words from another language (usually English)

### Example:

```
Original:
"सरकार ने नई योजना शुरू की"

Synthetic:
"Sarkar ne new scheme shuru ki"
```

### Rules:

* Replace only content words (nouns/adjectives)
* Keep sentence structure intact
* Do NOT fully translate sentence

---

## 🧠 Training Strategy

### Phase 1: Monolingual Training

* Train IndicBART on ILSUM dataset
* Goal: Learn summarization

---

### Phase 2: Code-Mixed Adaptation

Train on:

* Synthetic code-mixed data
* Real code-mixed dataset

Suggested mix:

* 70% synthetic
* 30% real

---

### Phase 3 (Optional but Recommended)

* Mix small portion of ILSUM again
* Prevents noisy generation

---

## 🏷️ Input Formatting

Use language tags:

```
<2hi> text...
<2ta> text...
<2en> text...
```

For code-mixed:

* Use **dominant language tag**

Example:

```
<2hi> kal meeting hai and we need to prepare report
```

---

## ⚙️ Training Configuration

* Model: IndicBART
* Epochs: 3–5 per phase
* Learning rate: 3e-5
* Batch size: 2–4 (use gradient accumulation)
* Max input length: 512–768 tokens
* Use mixed precision if possible

---

## 📏 Evaluation Metrics

### Mandatory:

* ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)

### Optional:

* BERTScore (for semantic similarity)

---

## 🚀 Inference Pipeline

```
Input (code-mixed text)
   ↓
Basic preprocessing
   ↓
(Optional) Transliteration
   ↓
Add language tag
   ↓
Tokenization (IndicBART tokenizer)
   ↓
Model generates summary
```

---

## 🧪 Baseline vs Final System

### Baseline:

* Train IndicBART on ILSUM only
* No synthetic data

### Final System:

* Add synthetic code-mixed data
* Fine-tune on real code-mixed dataset
* Improved performance on mixed input

---

## ⚠️ Challenges & Solutions

### 1. Code-mixed data scarcity

→ Use synthetic data generation

### 2. Noisy input (spelling, slang)

→ Keep preprocessing simple

### 3. Language confusion

→ Use language tags

### 4. Limited compute

→ Use small batch + gradient accumulation

---

## 📅 Timeline (1 Month Plan)

### Week 1:

* Learn transformers (HuggingFace)
* Setup environment
* Load datasets

### Week 2:

* Implement preprocessing
* Generate synthetic data

### Week 3:

* Train Phase 1 + Phase 2 models

### Week 4:

* Fine-tune + evaluate
* Improve results
* Prepare project/demo

---

## 💡 Key Innovations

* Multilingual **code-mixed summarization**
* Synthetic data generation for low-resource task
* Hybrid training (monolingual + code-mixed)
* Practical transformer-based pipeline

---

## 🧾 Tech Stack

* Python
* HuggingFace Transformers
* PyTorch
* Indic NLP tools (optional)
* Datasets: ILSUM + Hinglish/GupShup

---

## 🎯 Final Outcome

A working system that:

* Summarizes multilingual code-mixed text
* Handles noisy real-world input
* Demonstrates strong NLP + DL understanding
* Is suitable for **internships / research portfolios**
