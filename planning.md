# TakeMeter Planning

## Community

I chose Hacker News comments as the community for this classifier. Hacker News is a good fit because the same thread routinely contains developed technical explanations, first-person experience reports, and very short reactions or jokes, which makes the discourse varied enough to support a meaningful text classification task.

This distinction matters in this community because readers often want to separate substantial reasoning from anecdotal evidence and from drive-by reactions. A classifier that can reliably distinguish those modes of discourse could support moderation, ranking, or downstream analysis of how a thread evolves.

## Label Taxonomy

### `analysis`

Definition: The comment mainly explains, argues, or evaluates with concrete reasoning, mechanism, evidence, or comparison.

Clear examples:

1. "He actively campaigned against any regulation of derivatives. There is an infamous lunch that he had with Brooksley Born ... where she attempted to regulate them."
2. "IIRC linguists ... prefer to break up kana and analyze the consonant and vowel separately when dealing with conjugations."

Uncertain example:

"Really? 'Oh, someone I don't know! stab'? What if the person is plain-clothes law enforcement? ..."

Why it is uncertain: it is framed as a reaction, but the body of the comment works through counterexamples and therefore functions more like reasoning than pure reaction.

### `experience`

Definition: The comment mainly relies on the writer's own experience, project, job, or personal example to make its point.

Clear examples:

1. "Nice job! I made something similar, mostly for myself: https://www.vexling.com Plan to keep it forever free :)"
2. "Mine is currently Net, 7x7 grid with wrapping variant. I take about 5 minutes on average to solve a level, which is the sweet spot for me."

Uncertain example:

"First, this is true - learning or doing anything does change your brain. ... I personally don't think it's meant to scare-monger ..."

Why it is uncertain: the comment contains both personal stance and general reasoning. My rule is that if the first-person perspective is carrying the argument, it stays `experience`.

### `reaction`

Definition: The comment is primarily a quick reaction, joke, praise, dismissal, or unsupported opinion rather than developed reasoning.

Clear examples:

1. "Can we start a trend of wearing ski masks and other face coverings in public?"
2. "Yeah, that irked me a bit but maybe it was intentional?"

Uncertain example:

"Read the whole article expecting it to explain how it would have changed, was disappointed to not read that."

Why it is uncertain: it gestures toward a critique, but it does not develop the critique with evidence or mechanism, so it remains `reaction`.

## Hard Edge Cases

The hardest boundary is `analysis` vs `experience`. Hacker News commenters often start with "I think" or "I noticed" and then continue into a detailed argument. My decision rule is:

If the substance of the comment would still stand as an argument after removing the first-person framing, label it `analysis`.

If the first-person experience is the main evidence and the comment would lose most of its force without that personal context, label it `experience`.

Three real difficult cases from the dataset:

1. "Becoming a new parent I knew I'd sleep less ... what I didn't know was simultaneously how much energy I'd have to keep going." Decision: `analysis`. It starts personally, but the comment is trying to explain a broader pattern rather than merely report an anecdote.
2. "I agree 100% with the message ... but it feels like the horse is already out of the barn ..." Decision: `analysis`. Despite the first-person opening, the core move is a general argument about facial surveillance saturation.
3. "The article is saying don't ask for permission! ... they are saying it's common to ask permission in cases where you shouldn't ..." Decision: `experience`. This one could have been `analysis`, but I treat it as `experience` because the argument is grounded in the writer's reading of the conversation and own interpretive stance rather than in external evidence.

## Data Collection Plan

Source: public Hacker News comments fetched from the Algolia Hacker News API.

Collection approach:

1. Pull recent public comments in batches of 100.
2. Clean HTML out of the comment text.
3. Exclude very short comments under 40 characters.
4. Build a single CSV with `text`, `label`, and `notes`, plus provenance columns.

Target distribution:

1. 70 `analysis`
2. 70 `experience`
3. 70 `reaction`

If a label is underrepresented after 200 examples, I will keep collecting more comments from later API pages and oversample the scarce discourse type until the dataset is balanced enough to avoid a majority-class model. If one label becomes too rare in naturally occurring recent comments, I would shift collection toward thread types where that discourse mode appears more often, such as `Show HN` for `experience` and policy or language threads for `analysis`.

## Evaluation Metrics

I will use:

1. Accuracy, because the final classifier still needs a simple overall score.
2. Per-class precision, recall, and F1, because the three labels are not equally easy and I need to know whether one class is collapsing.
3. Macro F1, because it penalizes models that do well only on the easiest class.
4. A confusion matrix, because directional mistakes are central to this task. The main question is not only "how often is the model wrong" but "which discourse modes does it confuse with which others."
5. Baseline parse rate for the Groq prompt, because a zero-shot baseline is not useful if it frequently outputs labels in an unusable format.

Accuracy alone is not enough. A model could get a reasonable accuracy score by overpredicting `reaction`, since short reactive comments are common. Per-class metrics and the confusion matrix are needed to verify that the model actually learns the intended distinctions.

## Definition Of Success

This classifier would be genuinely useful if it meets all of the following on the held-out test set:

1. Fine-tuned accuracy of at least 0.65.
2. Macro F1 of at least 0.60.
3. No class F1 below 0.50.
4. Fine-tuned model beats the zero-shot baseline by at least 0.08 absolute accuracy.

For a real deployment in a community analysis tool, I would call the model good enough if it reliably separates `reaction` from the other two classes and keeps `analysis` vs `experience` confusion at a level that is understandable and diagnosable rather than random. If those thresholds are missed, the project still succeeds as an analysis exercise, but not as a deployable classifier.

## AI Tool Plan

### Label Stress-Testing

I will give the label definitions and edge-case rule to an AI assistant and ask it to generate 5 to 10 boundary cases, especially comments that mix first-person phrasing with generalized reasoning. If those synthetic cases cannot be classified consistently, I will tighten the `analysis` vs `experience` rule before doing more annotation.

### Annotation Assistance

I will use a lightweight rule-based pre-labeling pass to speed up collection, then review outputs manually before treating the labels as final. If I later use an LLM to pre-label additional comments, I will track that explicitly and disclose it in the README's AI usage section.

### Failure Analysis

After running the baseline and the fine-tuned model, I will paste wrong predictions into an AI assistant and ask for recurring themes such as sarcasm, short comments, or first-person framing. I will only keep patterns that I can verify by rereading the misclassified examples myself.
