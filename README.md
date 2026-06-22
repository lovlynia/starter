# AI201 Project 3: TakeMeter

## Status

This repository is prepared for the TakeMeter workflow except for the Colab-only execution steps that require your Google account session and Groq API secret. The dataset, planning document, starter script customization, and reporting structure are in place.

## Community Choice And Reasoning

This project studies Hacker News comments. Hacker News is a strong classification target because the same thread often mixes three distinct discourse modes: developed reasoning, first-person experience, and short reactive commentary. Those distinctions matter to readers because they affect how trustworthy, reusable, and discussion-driving a comment is.

## Label Taxonomy

### `analysis`

The comment mainly explains, argues, or evaluates with concrete reasoning, mechanism, evidence, or comparison.

Examples:

1. "He actively campaigned against any regulation of derivatives ..."
2. "IIRC linguists ... prefer to break up kana and analyze the consonant and vowel separately ..."

### `experience`

The comment mainly relies on the writer's own experience, project, job, or personal example to make its point.

Examples:

1. "Nice job! I made something similar, mostly for myself ..."
2. "Mine is currently Net, 7x7 grid with wrapping variant ..."

### `reaction`

The comment is primarily a quick reaction, joke, praise, dismissal, or unsupported opinion rather than developed reasoning.

Examples:

1. "Can we start a trend of wearing ski masks and other face coverings in public?"
2. "Yeah, that irked me a bit but maybe it was intentional?"

## Data Collection Source And Labeling Process

The dataset lives in `hn_hn_comments_labeled.csv` and contains 210 public Hacker News comments collected from the Algolia Hacker News API. The generator is in `scripts/generate_hn_dataset.py`.

Collection and labeling workflow:

1. Fetch recent public comments in pages of 100.
2. Strip HTML and normalize whitespace.
3. Drop comments shorter than 40 characters.
4. Apply a first-pass rule-based label using the taxonomy above.
5. Use the generated output as the single CSV that the TakeMeter notebook expects.

### Label Distribution

| Label | Count |
| --- | ---: |
| `analysis` | 70 |
| `experience` | 70 |
| `reaction` | 70 |

### Three Difficult-To-Label Examples

1. "Becoming a new parent I knew I'd sleep less ... what I didn't know was simultaneously how much energy I'd have to keep going." I treat this as `analysis` because it attempts a general explanation, not just a diary-style anecdote.
2. "I agree 100% with the message ... but it feels like the horse is already out of the barn ..." I treat this as `analysis` because the argumentative structure matters more than the first-person stance.
3. "Read the whole article expecting it to explain how it would have changed, was disappointed to not read that." I treat this as `reaction` because it expresses dissatisfaction without building a supported critique.

## Fine-Tuning Approach

Base model: `distilbert-base-uncased`

Planned training setup in the starter notebook:

1. Train/validation/test split of 70% / 15% / 15%.
2. Three epochs.
3. Learning rate `2e-5`.
4. Train batch size `16`.
5. Best checkpoint selected by validation accuracy.

Hyperparameter decision: I kept the notebook defaults because the dataset is small and balanced, so the main risk is overfitting rather than undertraining. The first change I would consider only after a failed run is lowering batch size to `8` if Colab memory becomes an issue.

## Baseline Description

The baseline in the customized starter script uses Groq with `llama-3.3-70b-versatile` and a zero-shot classification prompt. The prompt defines the Hacker News community, the three labels, one concrete example per label, and instructs the model to return only the label name.

Prompt source: see `ai201_project3_takemeter_hn.py`.

## Evaluation Report

This section is ready for the Colab outputs but cannot be completed truthfully from this environment because the run depends on your Colab session, T4 runtime selection, and Groq secret.

### Metrics Table

Fill this table after running Sections 4 through 6 in Colab.

| Model | Accuracy | Macro F1 | Notes |
| --- | ---: | ---: | --- |
| Groq zero-shot baseline | TBD | TBD | Add parseable-response count |
| Fine-tuned DistilBERT | TBD | TBD | Add comparison to baseline |

### Per-Class Metrics

Add the per-class precision, recall, and F1 tables for both models here after the Colab run.

### Fine-Tuned Confusion Matrix

Write the matrix out as a markdown table after the Colab run.

| True \ Pred | `analysis` | `experience` | `reaction` |
| --- | ---: | ---: | ---: |
| `analysis` | TBD | TBD | TBD |
| `experience` | TBD | TBD | TBD |
| `reaction` | TBD | TBD | TBD |

### Failure Analysis

Add at least three wrong predictions from the fine-tuned model here after the Colab run. Focus on whether the errors cluster around `analysis` vs `experience`, because that is the hardest semantic boundary in this dataset.

Recommended structure for each example:

1. Comment text.
2. True label.
3. Predicted label.
4. Confidence.
5. Why the model likely failed.
6. What change might reduce that failure mode.

### Sample Classifications

Add 3 to 5 examples after the Colab run, including one correct prediction that you explain as reasonable.

| Example Text | Predicted Label | Confidence | Commentary |
| --- | --- | ---: | --- |
| TBD | TBD | TBD | TBD |

## Reflection: What The Model Learned vs What I Intended

The intended distinction is between reasoning, experience, and reaction. The likely failure mode is that the model will overuse lexical cues such as first-person pronouns or short length instead of learning a deeper discourse-mode distinction. If the fine-tuned model confuses `analysis` and `experience` more than it confuses either with `reaction`, that would indicate the model learned the easy boundary first and only partially captured the more nuanced one.

## Spec Reflection

One way the spec helped is that it forced a hard label-boundary rule before large-scale annotation. That prevented the taxonomy from collapsing into vague notions like "good" and "bad" comments.

One way the implementation diverged from the original intent is that I used a reproducible API-backed collection script instead of a manual spreadsheet workflow. The reason is practical: it produces a documented, repeatable dataset build while still using public comments and the same label definitions.

## AI Usage

1. I used GitHub Copilot to help design and tighten the label taxonomy. It proposed boundary-sensitive discourse labels and I revised the initial rules to keep `analysis` and `experience` distinct.
2. I used GitHub Copilot to build the public-comment collection and pre-labeling script. The first pass mislabeled some short jokes as `analysis`; I corrected the rule so short unsupported comments fall into `reaction` unless they show clear reasoning markers.

If additional AI assistance is used during the Colab run, especially for failure analysis or any future pre-label review, add that here explicitly.

## Colab Run Checklist

1. Open the starter notebook in Colab and save a copy in Drive.
2. Set the runtime to T4 GPU.
3. Add `GROQ_API_KEY` in Colab Secrets.
4. Upload `hn_hn_comments_labeled.csv` in Section 1.
5. Use the filled label map and system prompt from `ai201_project3_takemeter_hn.py`.
6. Run Sections 1, 2, and 5 for the baseline.
7. Run Sections 3, 4, and 6 for fine-tuning and exports.
8. Download `evaluation_results.json` and `confusion_matrix.png` and add them to this repo.
