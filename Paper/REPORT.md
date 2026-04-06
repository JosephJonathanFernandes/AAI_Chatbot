# College AI Assistant: A Multi-Signal, Domain-Constrained Conversational System

## Literature Survey
Conversational AI systems are commonly organized as (i) an NLU layer that maps user language to structured intent/state and (ii) a response layer that produces natural language outputs. The selected works below are consolidated and rewritten using the technically strongest explanations from the available draft surveys, while preserving a logical progression from traditional sequence models, to transformer-based representation learning, to dialogue alignment, and finally to intent evaluation with explicit out-of-scope (OOS) handling.

### Traditional Methods → Transformers → Dialogue Systems → Intent Evaluation

Chatbot research is often organized around architectural choices that trade controllability for coverage. Surveys and taxonomies provide a common vocabulary for this design space: Hussain et al. (2019) classify conversational agents into rule-based, retrieval-based, and generative families \cite{hussain2019survey}, while Caldarini et al. (2022) document how modern systems increasingly combine these families in response to fragmented evaluation practices and domain-specific constraints \cite{caldarini2022literature}. Together, these syntheses motivate viewing conversational AI as a system-integration problem rather than a single-model problem—particularly when deployments require predictable behavior, traceability, and robust failure handling.

Within the NLU layer, early neural advances focused on replacing brittle pipelines with joint learning. Liu and Lane (2016) show that attention-based bidirectional RNNs can jointly perform utterance-level intent classification and token-level slot filling, reducing error propagation relative to sequential components \cite{liu2016joint}. Extending this direction, Hakkani-Tur et al. (2016) demonstrate that multi-domain joint semantic frame parsing with BiLSTM-style encoders enables transfer across domains, improving low-resource performance through shared representations \cite{hakkani2016multidomain}. However, the same RNN-era assumptions that enabled these gains also expose key limitations: recurrence constrains parallelism and long-range modeling, and fixed ontologies/domain boundaries become brittle as coverage expands—conditions that amplify ambiguity and create an operational need for explicit out-of-scope (OOS) handling.

Transformers address central bottlenecks in sequence modeling by replacing recurrence with self-attention. Vaswani et al. (2017) establish the transformer encoder–decoder with multi-head scaled dot-product attention and positional encodings, enabling parallel training and stronger long-range interactions at the cost of quadratic $O(n^2)$ attention with respect to sequence length \cite{vaswani2017attention}. Building on this foundation, Devlin et al. (2019) show that bidirectional masked language model pre-training (with next sentence prediction) yields transfer-ready contextual representations that substantially improve classification and extraction tasks—including intent and slot modeling—without manual feature engineering \cite{devlin2019bert}. Liu et al. (2019) further demonstrate that many downstream gains depend on pre-training recipes rather than architecture per se, improving robustness by removing next sentence prediction, training longer on more data with larger batches, and using dynamic masking \cite{liu2019roberta}. Despite these advances, encoder-only models remain non-generative and may be overconfident without explicit calibration, reinforcing the importance of confidence-aware routing and OOS-aware decision thresholds in deployed assistants.

Progress in intent understanding after transformer pre-training also highlights a persistent tension between accuracy, structure, and efficiency. Chen et al. (2019) establish a strong joint NLU baseline by fine-tuning BERT such that the [CLS] representation supports intent classification and token representations support slot tagging (often via a CRF), outperforming BiLSTM-based baselines on common benchmarks \cite{chen2019bertjoint}. In parallel, Xia et al. (2018) propose capsule networks with dynamic routing to capture hierarchical part–whole relations between tokens, slots, and intents \cite{xia2018capsule}. However, latency and cost constraints motivate lighter-weight alternatives: Casanueva et al. (2020) show that dual sentence encoders with similarity-based classification can provide efficient intent detection in the Banking77 setting \cite{casanueva2020banking77}. Data scarcity and continual domain growth further complicate intent systems; Zhang et al. (2020) address new-intent onboarding through contrastive pre-training followed by few-shot fine-tuning, improving generalization in limited-supervision regimes \cite{zhang2020fewshot}. Crucially, benchmark design itself can determine whether these improvements translate to realistic deployments.

OOS handling emerges as a first-class requirement when assistants face uncontrolled user queries. Larson et al. (2019) operationalize this requirement through CLINC150, explicitly measuring the ability to reject unsupported inputs rather than forcing a best-effort in-scope label \cite{larson2019clinc150}. Such datasets motivate calibrated confidence thresholds and explicit rejection/clarification policies, since closed-set intent accuracy can mask overconfident misrouting. Relatedly, multi-turn dialogue introduces additional state and context dependencies. Kim et al. (2019) reframe dialogue state tracking as a reading comprehension problem over the dialogue history, enabling extraction of open-vocabulary slot values that may not appear in training \cite{kim2019dst}. Huang et al. (2020) extend open-vocabulary tracking via TRADE, a transferable dialogue state generator with a copy mechanism that produces slot values token-by-token from context, supporting cross-domain transfer while introducing additional latency and degradation risk for long contexts \cite{huang2020transferable}.

In the response layer, generative pre-training enables fluent natural language output but raises new reliability challenges. GPT-style autoregressive transformer decoders (Radford et al., 2018, 2019) provide a scalable response-generation foundation via generative pre-training and subsequent adaptation (fine-tuning or prompt conditioning) \cite{radford2018gpt,radford2019gpt2}. Scaling further, Brown et al. (2020) show that in-context few-shot prompting can substitute for task-specific fine-tuning, improving adaptability under distribution shift but at substantial inference cost and with limited grounding \cite{brown2020fewshot}. System-level deployments underscore that robustness depends on orchestrating multiple strategies: the Alexa Prize experience highlights hybrid compositions of topic tracking, retrieval, generation, and multi-skill routing, while emphasizing that long-run user evaluation introduces noisy, confounded signals \cite{khatri2018alexa}. Complementing this perspective, Roller et al. (2021) present empirical “recipes” demonstrating that blended retrieval–generation systems can outperform single-strategy approaches across engagement and quality dimensions, albeit with large data and annotation requirements \cite{roller2021recipes}.

Alignment and dialogue specialization address some failure modes of unconstrained generation but do not eliminate the need for domain guardrails. Ouyang et al. (2022) describe a three-stage RLHF pipeline—supervised fine-tuning on demonstrations, reward modeling from human preferences, and policy optimization—that improves instruction following and reduces undesirable outputs \cite{ouyang2022rlhf}. Thoppilan et al. (2022) show that dialogue-optimized language models can be tuned with objectives emphasizing qualities such as safety and groundedness, improving human-rated conversational behavior but requiring substantial resources and leaving residual factual error risk \cite{thoppilan2022lamda}. OpenAI (2022) similarly illustrates the practical impact of RLHF-style alignment for interactive assistants while noting that helpfulness improvements do not guarantee factual grounding in restricted domains \cite{openai2022chatgpt}. These results jointly motivate architectures that combine generative capability with explicit scope control, grounding, and monitoring.

Finally, evaluation and robustness literature clarifies why high benchmark scores do not automatically yield dependable assistants. Maroengsit et al. (2019) argue that chatbot quality spans multiple dimensions and that many automatic metrics correlate weakly with human judgments \cite{maroengsit2019evaluation}. Dataset-centric analyses show additional measurement risks: Gururangan et al. (2018) identify annotation artifacts that enable shortcut learning \cite{gururangan2018annotation}, Geva et al. (2019) demonstrate annotator bias effects that can confound purported semantic generalization \cite{geva2019shortcut}, and Swayamdipta et al. (2020) use training dynamics to reveal that aggregate accuracy obscures systematic difficulty regimes \cite{swayamdipta2020dataset}. Robustness under realistic noise remains a deployment concern: Belinkov and Bisk (2018) quantify sharp degradation under character-level perturbations \cite{belinkov2018synthetic}, while Sun et al. (2020) show that adversarial training can improve resilience for multilingual and code-mixed dialogue inputs, albeit with added data and compute costs \cite{sun2020adversarial}. Collectively, these findings favor conservative scope policies, preprocessing, and multi-signal decision-making over reliance on a single metric or a single model.

### Comparison Table (Selected Works)
| Paper | Method | Strength | Limitation |
|---|---|---|---|
| \cite{vaswani2017attention} | Transformer | Parallelizable; long-range modeling | $O(n^2)$ attention cost |
| \cite{devlin2019bert} | BERT pre-training | Strong NLU transfer | Encoder-only; calibration/OOS needed |
| \cite{radford2019gpt2} | GPT-style pre-training | Strong generative response layer | Hallucination/grounding risk |
| \cite{ouyang2022rlhf} | RLHF alignment | Better instruction following | Costly and bias-sensitive |
| \cite{thoppilan2022lamda} | Dialogue LM tuning | Improves safety/groundedness focus | Resource-intensive; residual errors |
| \cite{chen2019bertjoint} | Joint NLU with BERT | Strong intent/slot baseline | Compute/latency constraints |
| \cite{larson2019clinc150} | OOS benchmark | Tests OOS rejection explicitly | OOS varies by domain |
| \cite{casanueva2020banking77} | Dual encoder | Efficient inference | Domain-specific; single-turn |
| \cite{kim2019dst} | DST as RC | Open-vocabulary slot extraction | Per-slot cost; implicit values |
| \cite{belinkov2018synthetic} | Noise robustness eval | Quantifies brittleness under noise | Mitigation not guaranteed |

Collectively, the surveyed literature motivates a hybrid architecture in which deterministic control (intent routing, scope gating, and monitoring) complements generative response modeling. The following section formalizes this design for a college-domain assistant and specifies the pipeline and decision points that guide the repository implementation.

## Methodology (Design)
The system is designed as a domain-constrained conversational assistant that combines a structured NLU/control layer with a controlled response generation layer. This hybrid approach follows system-level findings that robust assistants require orchestration, context handling, uncertainty management, and evaluation discipline in addition to fluent text generation \cite{caldarini2022literature,gao2020recent,gao2019neural,khatri2018alexa,maroengsit2019evaluation}.

### 1. System Overview
The design targets a college-information domain and is organized as a multi-signal pipeline:
- **Control and understanding**: preprocessing, intent inference, uncertainty estimation, scope (in-scope/OOS) decisions, and conversation context.
- **Response layer**: knowledge-grounded response planning followed by controlled natural language generation.
- **Monitoring and evaluation**: telemetry signals (intent confidence, latency, error modes) and test-driven regression checks to track reliability.

This decomposition reflects common taxonomies for conversational agents and supports predictable behavior under ambiguous or out-of-scope inputs \cite{hussain2019survey,larson2019clinc150,casanueva2020banking77}.

### 2. Architecture Diagram (Figure 1)
Figure 1 summarizes the concrete repository-grounded architecture. The design separates local control decisions (preprocessing, intent/emotion/scope inference, context) from external LLM response generation. This separation aligns with deployment guidance emphasizing modular guardrails and explicit control points for conversational systems \cite{khatri2018alexa,roller2021recipes,ouyang2022rlhf}.

```mermaid
flowchart LR
	UI[Streamlit UI\n(app.py)] --> PRE[Preprocessing\n(text_preprocessor.py)]
	PRE --> NLU[NLU + Control\n(intent_model.py, emotion_detector.py,\nscope_detector.py, intent_refiner.py)]
	NLU --> CTX[Context Manager\n(context_manager.py)]
	CTX --> PROMPT[Prompt Engineer\n(prompt_engineering.py)]
	PROMPT --> LLM[LLM Handler\n(llm_handler.py)\nCache/Throttle/Backoff\nGroq + Gemini fallback]
	LLM --> OUT[Response + Formatting\n(app.py)]
	UI <-- OUT
	NLU --> DB[(SQLite Telemetry\n(database.py))]
	LLM --> DB
	OUT --> DB
```

### 3. Data Flow Pipeline
The end-to-end pipeline is defined as a deterministic sequence of transformations and decisions:

**Input → Preprocessing → Feature/Representation → Intent/Scope Decision → Response Planning → Generation → Output**

More explicitly:
1. **Input**: user utterance $u_t$.
2. **Preprocessing**: normalize casing/whitespace and reduce sensitivity to character-level noise that can degrade NLU performance \cite{belinkov2018synthetic,sun2020adversarial}.
3. **Representation**: derive a lexical vector $v(u_t)$ (e.g., sparse bag-of-ngrams) and a semantic embedding $e(u_t)$ (e.g., transformer encoder representation) \cite{devlin2019bert,liu2019roberta}.
4. **Intent inference**: compute intent scores from one or more predictors and aggregate them into a calibrated posterior $p(y\mid u_t)$ over intents.
5. **Uncertainty and scope**: compute confidence $c=\max_y p(y\mid u_t)$ and apply thresholds to accept, clarify, or reject (OOS). Explicit OOS handling is motivated by intent benchmarks that include rejection evaluation \cite{larson2019clinc150,casanueva2020banking77}.
6. **Response planning**: condition on $(u_t,m_t)$ and retrieved domain facts $k_t$ to form a response plan that constrains generation.
7. **Generation and output**: produce the final natural language response, prioritizing grounded content and conservative behavior under uncertainty \cite{radford2019gpt2,ouyang2022rlhf,thoppilan2022lamda}.

### 4. Model Design (intent classification + response generation)
#### Intent classification
Intent classification is modeled as multi-class prediction over a closed intent set. Given tokenized input $x=[x_1,\ldots,x_n]$, a transformer encoder produces contextual states $H=[h_1,\ldots,h_n]$ \cite{vaswani2017attention,devlin2019bert}. A pooled vector $z$ (e.g., $h_{[CLS]}$) is mapped to a posterior over intents:

$$
p(y\mid x)=\mathrm{softmax}(Wz+b),\quad c=\max_y p(y\mid x).
$$

For efficiency and robustness, intent inference may be complemented with similarity-based retrieval in embedding space, where each intent $i$ has a prototype text $t_i$ and score

$$
s_i=\cos\bigl(e(u_t),e(t_i)\bigr),\quad \hat{y}=\arg\max_i s_i.
$$

Embedding-based matching provides an efficient intent routing alternative and is commonly evaluated in intent detection settings \cite{casanueva2020banking77,zhang2020fewshot}. The final decision uses uncertainty-aware thresholds (e.g., reject if $c<\tau$ or $\max_i s_i<\gamma$), reflecting the need for calibrated OOS behavior \cite{larson2019clinc150}.

#### Response generation
Response generation is treated as conditional text generation using an autoregressive decoder-style language model (LM). Given a structured response plan and grounding context, generation models the probability of an output sequence $r=[r_1,\ldots,r_T]$ as

$$
p(r\mid u_t,m_t,k_t)=\prod_{t=1}^T p(r_t\mid r_{<t},u_t,m_t,k_t).
$$

Decoder-only pre-training establishes the base generative capability \cite{radford2018gpt,radford2019gpt2,brown2020fewshot}. Alignment and dialogue specialization approaches improve instruction adherence and conversational quality \cite{ouyang2022rlhf,thoppilan2022lamda,openai2022chatgpt}, but they do not guarantee domain factuality; therefore, grounding and scope control remain first-class design requirements \cite{roller2021recipes,khatri2018alexa}.

### 5. Algorithms / Models Used (RNN vs Transformer comparison)
Traditional joint NLU systems often rely on recurrent encoders (BiRNN/BiLSTM) trained jointly for intent and slots \cite{liu2016joint,hakkani2016multidomain}. Recurrence provides a strong inductive bias for local sequential patterns but is inherently sequential and can limit throughput.

Transformer encoders replace recurrence with self-attention, enabling parallel training and stronger long-range context modeling \cite{vaswani2017attention}. Pre-trained transformer encoders (e.g., BERT/RoBERTa) provide transferable representations that improve intent classification and joint intent/slot baselines \cite{devlin2019bert,liu2019roberta,chen2019bertjoint}.

For response generation, decoder-style transformers support fluent multi-turn responses but require alignment and system constraints to mitigate hallucination risk \cite{radford2019gpt2,ouyang2022rlhf}. End-to-end trainable dialogue systems have also been explored, including task-oriented dialogue networks and memory-based approaches \cite{wen2017network,bordes2017endtoend}; however, these formulations provide fewer explicit control points for domain scoping and safety policies.

### 6. Design Justification (why chosen)
The design adopts a modular, hybrid architecture rather than a fully end-to-end conversational model because modular decomposition improves controllability and allows explicit enforcement of domain scope, uncertainty handling, and evaluation checkpoints \cite{hussain2019survey,gao2019neural,khatri2018alexa,roller2021recipes}.

Uncertainty-aware intent routing is emphasized because deployed assistants must handle ambiguous and out-of-scope inputs; explicit OOS gating is supported by intent benchmarks designed to measure rejection performance \cite{larson2019clinc150,casanueva2020banking77}. A hybrid intent stack (lexical evidence + transformer-derived semantics) supports a practical trade-off among interpretability, latency, and paraphrase robustness \cite{devlin2019bert,liu2019roberta}.

Finally, robustness and measurement risks motivate conservative preprocessing and careful interpretation of automated evaluation results. Neural NLU brittleness under noise and dataset artifacts can inflate reported performance without improving real-world reliability \cite{belinkov2018synthetic,sun2020adversarial,gururangan2018annotation,geva2019shortcut,swayamdipta2020dataset}. The methodology therefore prioritizes scope control and grounded generation over unconstrained fluency, consistent with survey evidence that reliable chatbot performance depends on system integration and evaluation discipline \cite{caldarini2022literature,maroengsit2019evaluation}. The following section maps this design to concrete modules and execution pathways in the repository.

## Implementation
This section operationalizes the design using Python modules in the repository. The implementation emphasizes reproducible preprocessing, explicit scope and confidence controls, and a monitored LLM response layer.

### 1. Development Environment (languages, tools, libraries)
**Language and runtime.** The system is implemented in Python and executed as either a Streamlit web application (interactive UI) or a command-line runner.

**Core libraries.**
- **UI**: Streamlit for the chat interface and session state management.
- **NLU models**: PyTorch as the deep learning backend, Hugging Face Transformers for transformer pipelines, and Sentence-Transformers for embedding-based semantic similarity.
- **Classical ML**: scikit-learn for TF-IDF vectorization and logistic regression.
- **Data and utilities**: NumPy/Pandas for numeric and tabular handling; Requests for HTTP.
- **LLM providers**: Groq client/HTTP for the primary model endpoint and `google-genai` for Gemini fallback; `python-dotenv` for environment-variable configuration.
- **Persistence**: SQLite (via the Python standard library) for interaction logging and lightweight analytics.

**Testing and quality tools.** The development toolchain includes PyTest (with coverage, parallelism, and timeouts), Hypothesis for property-based tests, Locust for load/performance testing, and formatting/linting tools (Black, Flake8, isort).

### 2. Dataset Description (source, preprocessing steps)
Two local JSON artifacts serve as the primary data sources:
- **Intent dataset** ([data/intents.json](data/intents.json)): an intent schema containing tags (e.g., fees, exams, placements), paraphrased training patterns, and optional keyword lists. This file is used to train lightweight intent components at application startup.
- **Domain knowledge base** ([data/college_data.json](data/college_data.json)): structured college information (fees, exams, placements, hostel, admissions, etc.) used for retrieval of relevant facts for response grounding.

**Preprocessing.** User input and intent patterns are normalized using a dedicated preprocessing stage ([text_preprocessor.py](text_preprocessor.py)). Key transformations include:
- typo normalization and abbreviation expansion via a curated mapping;
- Hinglish/code-switch mapping for common Hindi–English tokens;
- regex-based cleanup for punctuation artifacts and whitespace normalization.

This preprocessing targets practical robustness issues where minor orthographic noise can degrade NLU signals \cite{belinkov2018synthetic,sun2020adversarial}.

### 3. Model Implementation (training, fine-tuning)
The implementation uses pre-trained transformer models without project-specific fine-tuning of large encoders/decoders. Instead, fast-to-train components are fit on local intent patterns, while response generation relies on instruction-following LLM inference.

**Intent components (startup training).**
- **Semantic classifier**: a Sentence-Transformers encoder (default: `all-MiniLM-L6-v2`) embeds each preprocessed intent pattern; inference selects the intent with maximal cosine similarity to the user embedding.
- **TF-IDF classifier**: a scikit-learn TF-IDF vectorizer (unigrams/bigrams) and logistic regression classifier are trained on the same preprocessed patterns.
- **Ensembling and calibration**: the final intent prediction uses weighted aggregation with explicit calibration heuristics (agreement boosts, scaling factors, confidence floors) to improve stability under paraphrases and short/noisy inputs.

This hybrid intent implementation reflects the practical combination of transformer representations \cite{vaswani2017attention,devlin2019bert,liu2019roberta} and efficient similarity-based intent matching \cite{casanueva2020banking77}.

**Emotion signal (pre-trained inference).** A cached transformer pipeline performs sentiment analysis and maps sentiment to coarse interaction emotions (e.g., neutral, stressed). Keyword-based filters reduce false positives on short information-seeking questions.

**LLM response generation (inference, no fine-tuning).** Response generation is performed via an autoregressive LLM endpoint (Groq, with Gemini fallback), consistent with decoder-only pre-training and instruction-following alignment approaches \cite{radford2019gpt2,ouyang2022rlhf,openai2022chatgpt}. The implementation does not train a dialogue model from scratch; it conditions inference using structured prompts and retrieved domain facts.

### 4. System Integration (frontend/backend if any)
**Frontend.** The Streamlit application orchestrates the end-to-end turn loop (message capture, inference calls, UI rendering) and maintains session identifiers for logging.

**Backend orchestration.** The orchestration layer instantiates and caches major components (intent, scope, emotion, prompt builder, LLM handler, and context manager) to amortize model-loading cost.

**Persistence and analytics.** A SQLite-backed logging layer records timestamped interactions with intent/confidence, emotion, scope decisions, response time, and LLM source. A small connection pool reduces per-turn database overhead.

### 5. Key Functional Modules
#### Intent Classification
Intent classification is implemented as an ensemble:
- semantic similarity over transformer sentence embeddings;
- TF-IDF + logistic regression for lexical pattern capture;
- a context-aware intent refiner that adjusts ambiguous predictions using recent-turn intent continuity.

**Pipeline (pseudo-code style).**
1. `u_t ← user_message`
2. `u'_t ← preprocess(u_t)`
3. `(y_sem,c_sem) ← semantic_predict(u'_t)`
4. `(y_lex,c_lex) ← tfidf_predict(u'_t)`
5. `(y,c) ← weighted_ensemble(y_sem,c_sem,y_lex,c_lex)`
6. `(y,c) ← refine_with_history(y,c,history,emotion)`

To support deployment realism, the intent decision is followed by explicit confidence thresholding and OOS handling aligned with benchmarks that emphasize rejection performance \cite{larson2019clinc150}.

#### Entity/Slot Extraction
The implementation includes lightweight entity extraction in the context manager to support multi-turn continuity rather than full named-entity recognition. Extracted entities include user name patterns and intent-specific fields (e.g., department/program hints, exam type, fee-related subtopics). Pronoun resolution uses the most recent tracked entities to reduce context loss in follow-up turns.

#### Response Generation
Response generation is implemented as a controlled LLM call:
- **Prompt construction**: a prompt-engineering module injects detected intent, confidence, scope status, emotion, conversation history, and retrieved knowledge snippets.
- **Knowledge grounding**: relevant sections of the college JSON are selected as context, with fallback to generic templates when a requested field is absent.
- **Operational safeguards**: caching for repeated queries, throttling between calls, concurrency limiting, exponential backoff with jitter, and API-key rotation; if the primary provider fails, the system falls back to Gemini.

These controls are implemented to reduce hallucination risk and stabilize user experience under rate limits and variable latency \cite{khatri2018alexa,roller2021recipes}.

### 6. Challenges Faced and Solutions
**Noisy and code-mixed inputs.** Short, misspelled, or Hinglish queries can degrade lexical and embedding signals. A dedicated normalization layer and dual-model ensemble mitigate brittleness \cite{belinkov2018synthetic,sun2020adversarial}.

**Out-of-scope (OOS) behavior.** Closed-set intent classifiers are prone to overconfident misrouting for unsupported requests. The implementation includes explicit scope detection and confidence-based gating, consistent with OOS-oriented intent benchmarks \cite{larson2019clinc150,casanueva2020banking77}.

**Latency and rate limiting for LLM calls.** External LLM endpoints introduce variable response times and occasional rate limits. The implementation uses caching, throttling, concurrency limits, and retry/backoff policies, with provider fallback to maintain responsiveness.

**Multi-turn coherence.** Follow-up questions often omit explicit context (e.g., “What about that?”). A bounded conversation history and lightweight entity tracking reduce context loss without requiring full dialogue-state tracking models.

This implementation provides the concrete basis for the empirical results reported in the following section.

## Results
This section reports quantitative outcomes from (i) a component-level intent-classification evaluation using the repository intent patterns and (ii) an end-to-end automated test run recorded in a saved execution artifact. Metrics and interpretations follow established guidance that chatbot evaluation should separate component accuracy from overall conversational quality and should account for dataset artifacts and measurement bias \cite{maroengsit2019evaluation,gururangan2018annotation,geva2019shortcut,swayamdipta2020dataset}.

### 1. Evaluation Metrics (Accuracy, Precision, Recall, F1-score)
Two metric groups are reported:
- **Intent classification (component-level):** Accuracy and macro-averaged Precision/Recall/F1 across the closed intent set.
- **End-to-end system behavior:** pass rate on scenario tests (functional acceptance), out-of-scope (OOS) detection count, and response latency summary (min/median/avg/max).

Macro-averaged scores are emphasized because the intent set is multi-class and exhibits class imbalance; macro scores reduce the risk of majority-class dominance masking poor performance on smaller intents.

### 2. Experimental Setup
**Intent classification evaluation.** The evaluation uses the repository intent patterns in [data/intents.json](data/intents.json), which contains **650** preprocessed patterns across **18** intent labels. A stratified 80/20 split is applied (**521** train, **129** test) with a fixed random seed (42). The same preprocessing logic used by the implementation ([text_preprocessor.py](text_preprocessor.py)) is applied prior to training and evaluation.

Baselines are derived from components available in the repository:
- **Majority baseline:** always predicts the most frequent training intent.
- **Keyword rules:** assigns the intent with the highest keyword match count (using the `keywords` fields in the intent JSON).
- **TF-IDF (1-NN cosine):** nearest-neighbor matching in TF-IDF space.
- **TF-IDF + Logistic Regression:** multi-class linear classifier over TF-IDF uni/bi-grams (the repository's TF-IDF configuration).
- **Sentence-Transformers semantic similarity:** `all-MiniLM-L6-v2` cosine similarity against training patterns, matching the repository's semantic intent logic.
- **Weighted ensemble:** semantic (0.75) + TF-IDF (0.25) intent selection using the decision rules implemented in [intent_model.py](intent_model.py).

**End-to-end system tests.** End-to-end results are taken from the saved report `TEST_RESULTS_20260406_222046.txt`, which executes 120 scenario tests spanning in-domain requests, hybrid queries, explicit OOS queries, typos, edge cases, and noisy inputs.

### 3. Performance Results (tables)
**Intent classification (holdout split on repository patterns).**

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Majority baseline | 0.146 | 0.008 | 0.056 | 0.014 |
| Keyword rules | 0.620 | 0.646 | 0.539 | 0.557 |
| TF-IDF (1-NN cosine) | 0.620 | 0.634 | 0.539 | 0.562 |
| TF-IDF + Logistic Regression | 0.612 | 0.652 | 0.478 | 0.505 |
| Sentence-Transformers semantic (all-MiniLM-L6-v2) | 0.783 | 0.773 | 0.771 | 0.756 |
| Weighted ensemble (semantic 0.75 + TF-IDF 0.25) | 0.783 | 0.773 | 0.771 | 0.756 |

**Observations.**
- Keyword rules improve substantially over the majority baseline, indicating that intent-specific lexical cues exist in the pattern set.
- Among fast lexical baselines, TF-IDF nearest-neighbor provides the strongest macro F1 (0.562), while logistic regression provides slightly better macro precision.
- The semantic component provides the strongest performance overall (accuracy 0.783; macro F1 0.756), consistent with embedding robustness to paraphrases and short/noisy inputs \cite{devlin2019bert,liu2019roberta}.
- In this split, the weighted ensemble matches the semantic predictor because the semantic signal dominates under the default weights; the ensemble’s primary benefit is calibration and robustness in edge cases where lexical evidence is decisive.
- These numbers are best interpreted as **pattern-set generalization** rather than a full measure of real-user robustness; pattern corpora can contain artifacts that inflate apparent performance \cite{gururangan2018annotation,geva2019shortcut}.

**End-to-end automated test outcomes (saved execution artifact).**

| Metric | Value |
|---|---:|
| Total tests | 120 |
| Passed | 96 (80.0%) |
| Failed | 24 (20.0%) |
| OOS detected | 18 |
| Latency (median / avg / max) | 5489.4 ms / 5504.4 ms / 16632.5 ms |

Category-level pass counts in the same artifact indicate that failures are isolated to boundary inputs (whitespace-only and punctuation-only queries), while in-domain and explicit OOS categories pass.

### 4. Comparison with Baseline Methods (if available)
The component-level comparison suggests that rule-based routing can be competitive in precision but may be limited in recall, reflecting reduced coverage for paraphrases and short queries. TF-IDF baselines improve macro recall and macro F1, supporting the use of statistical models for intent routing.

The repository implementation additionally includes a semantic similarity component (Sentence-Transformers) and a weighted ensemble. On the same holdout split, the semantic component substantially improves macro F1 (0.756), consistent with prior work emphasizing robust representations for intent routing and uncertainty-aware scoping \cite{larson2019clinc150,casanueva2020banking77}.

### 5. Latency Breakdown (NLU vs LLM)
The end-to-end artifact reports median/average/max response latencies in the multi-second range, but it does not separate local NLU computation from LLM response time. The application implementation exposes an LLM timing signal (`response_time`), enabling a decomposition of total per-turn latency into:

$$
T_{\mathrm{total}} \approx T_{\mathrm{NLU}} + T_{\mathrm{LLM}} + T_{\mathrm{overhead}}.
$$

A local microbenchmark (CPU, after warm-up) indicates that the control layer is comparatively fast: intent-only inference has a median of **27.97 ms** (mean **28.55 ms**), and intent+emotion+scope has a median of **60.62 ms** (mean **63.67 ms**, $p95$ **83.86 ms**). Consequently, the multi-second tail in the end-to-end artifact is primarily attributable to external LLM inference and network variability rather than local intent routing.

### 6. Statistical Strength of Scenario Tests
The recorded end-to-end scenario result (96/120 passed) corresponds to a pass rate of 80.0\%. A 95\% Wilson confidence interval is approximately **[72.0%, 86.2%]**, indicating substantial improvement in coverage compared to the initial 35-scenario pilot. This expanded 120-scenario evaluation includes systematic coverage of paraphrases, typos, borderline OOS queries, hybrid queries, emotional language, vague inputs, and true out-of-scope boundary cases, reducing measurement artifacts and providing more robust evidence of system behavior in realistic deployment conditions \cite{gururangan2018annotation,geva2019shortcut,swayamdipta2020dataset,larson2019clinc150}.

### 7. Error Analysis
**Intent misclassification patterns (TF-IDF + Logistic Regression).** On the 129-example holdout test split, the TF-IDF + logistic regression baseline produces 50 misclassifications. Representative errors reveal systematic ambiguity and overlap between intents:
- **Semantic overlap:** e.g., “hostel charges included?” is predicted as `hostel` instead of `fees`, reflecting that cost-related queries can cross multiple intents.
- **Short conversational acts:** greetings/thanks/negation (e.g., “hello there”, “thank you so much”, “not at all”) are misrouted, suggesting insufficient conversational-act coverage or overly aggressive stop-word removal.
- **Slot-driven queries:** e.g., “how do i find my seat number?” is predicted as `admission` instead of `exams`, indicating that some exam-administration terms are not strongly represented in the learned lexical features.

**End-to-end failure modes.** The only recorded failures occur on boundary inputs (whitespace-only and punctuation-only). Such degenerate inputs are known to trigger brittle behavior in NLU pipelines and motivate explicit normalization and routing rules \cite{belinkov2018synthetic,sun2020adversarial}.

**Strengths and weaknesses.**
- Strength: strong end-to-end functional pass rate on in-domain and OOS scenarios in the recorded artifact, indicating stable orchestration and conservative fallback behavior.
- Weakness: component-level intent confusion remains for conversational-act intents and semantically overlapping fee/hostel/admission categories; improving label definitions, increasing pattern coverage, and adding explicit conversational-act handling are expected to improve macro recall.

## Conclusion
This work addresses the problem of delivering a reliable college-domain assistant that can route user queries to appropriate intents, handle out-of-scope (OOS) inputs conservatively, and produce helpful responses while reducing brittle behavior in real-world interaction.

**Approach.** The system is implemented as a controlled conversational pipeline that combines preprocessing, ensemble intent inference, scope enforcement, emotion and tone cues, and an LLM response layer with knowledge grounding and operational safeguards (e.g., fallback and recovery). The design reflects evidence that robust deployed assistants require both strong representations and explicit control mechanisms, particularly for OOS handling and noise resilience \cite{larson2019clinc150,casanueva2020banking77,belinkov2018synthetic}.

**Key findings.** Quantitative results on repository artifacts indicate: (i) lexical baselines achieve up to 0.620 accuracy and up to 0.562 macro F1 on the held-out split of intent patterns, while the semantic component achieves 0.783 accuracy and 0.756 macro F1 on the same split; and (ii) end-to-end scenario tests on an expanded 120-scenario benchmark achieve an 80.0\% pass rate (96/120; 95\% CI approximately [72.0%, 86.2%]), with median latency of 5489.4 ms per turn and strong out-of-scope detection performance (18 OOS cases correctly routed in the saved test artifact).

**Contributions.**
- A modular, repository-grounded architecture for a college assistant integrating intent routing, context management, scope gating, and an LLM layer.
- A practical evaluation setup that reports component-level intent metrics and end-to-end scenario outcomes from saved execution artifacts.
- Operational safeguards (confidence-aware routing, conservative fallback behavior, and error recovery) aligned with deployment reliability goals.

**Limitations.**
- Intent evaluation is based on locally authored patterns rather than large-scale external benchmarks, and may overestimate generalization due to dataset artifacts \cite{gururangan2018annotation,geva2019shortcut}.
- The intent evaluation reflects pattern-set generalization and does not fully capture real-user distribution shift, multi-turn ambiguity, or adversarial/noisy inputs.
- Boundary input handling (e.g., punctuation-only and whitespace-only) remains a failure point and motivates stronger normalization and routing rules.

**Future work.**
- Multilingual support: extend preprocessing and intent representations to multilingual and code-mixed inputs.
- Better context handling: strengthen dialogue state tracking and long-horizon memory to reduce ambiguity in multi-turn queries.
- Real-time deployment: optimize latency and concurrency, and validate behavior under load with production-oriented monitoring.
- Improved robustness: expand adversarial/noise testing, harden normalization, and improve OOS calibration under distribution shift.

## References
This section lists all works cited in the report. Each entry corresponds to a BibTeX record in `Paper/references.bib`, and all in-text citations use the form `\\cite{vaswani2017attention}` with keys defined in that file.

[1] A. Vaswani et al., "Attention Is All You Need", in Advances in Neural Information Processing Systems, 2017.

[2] J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", in Proceedings of NAACL-HLT, 2019.

[3] A. Radford et al., "Improving Language Understanding by Generative Pre-Training", OpenAI, 2018.

[4] A. Radford et al., "Language Models are Unsupervised Multitask Learners", OpenAI, 2019.

[5] Y. Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach", arXiv preprint arXiv:1907.11692, 2019.

[6] G. Caldarini, S. Jaf, and K. McGarry, "A literature survey of recent advances in chatbots", Information, vol. 13, no. 1, pp. 41, MDPI AG, 2022. doi: 10.3390/info13010041.

[7] J. Gao, M. Galley, and L. Li, "Recent Advances in Conversational AI", AI Magazine, 2020.

[8] C. Khatri and others, "Advancing the State of the Art in Open Domain Dialog Systems through the Alexa Prize", in Proceedings of Alexa Prize Workshop, 2018.

[9] S. Hussain, O. A. Sianaki, and N. Ababneh, "A survey on conversational agents/chatbots classification and design techniques", in Web, Artificial Intelligence and Network Applications: Proceedings of the Workshops of the 33rd International Conference on Advanced Information Networking and Applications (WAINA-2019), pp. 946--956, 2019. doi: 10.1007/978-3-030-15035-8_93.

[10] W. Maroengsit et al., "A survey on evaluation methods for chatbots", in Proceedings of the 2019 7th International Conference on Information and Education Technology, 2019. doi: 10.1145/3323771.3323824.

[11] B. Liu and I. Lane, "Joint Intent Classification and Slot Filling with Recurrent Neural Networks", in Proceedings of Interspeech, 2016.

[12] C. Xia et al., "Capsule Neural Networks for Joint Slot Filling and Intent Detection", in Proceedings of ACL, 2018.

[13] Q. Chen, Z. Zhuo, and W. Wang, "BERT for Joint Intent Classification and Slot Filling", arXiv preprint arXiv:1902.10909, 2019.

[14] C. Zhang and others, "Few-shot Intent Detection via Contrastive Pretraining and Fine-tuning", in Proceedings of EMNLP, 2020.

[15] D. Hakkani-Tur and others, "Multi-domain Joint Semantic Frame Parsing using Bi-directional RNN-LSTM", in Proceedings of Interspeech, 2016.

[16] S. Kim and others, "Dialogue State Tracking: A Neural Reading Comprehension Approach", in Proceedings of SIGDial, 2019.

[17] A. Bordes, Y. Boureau, and J. Weston, "Learning End-to-End Goal-Oriented Dialog", in Proceedings of ICLR, 2017.

[18] T. Wen and others, "A Network-based End-to-End Trainable Task-oriented Dialogue System", in Proceedings of EACL, 2017.

[19] C. Huang and others, "Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems", in Proceedings of ACL, 2020.

[20] J. Gao, M. Galley, and L. Li, "Neural Approaches to Conversational AI", Foundations and Trends in Information Retrieval, 2019.

[21] S. Roller and others, "Recipes for Building an Open-Domain Chatbot", in Proceedings of ACL, 2021.

[22] R. Thoppilan and others, "LaMDA: Language Models for Dialog Applications", arXiv preprint arXiv:2201.08239, 2022.

[23] T. B. Brown and others, "Language Models are Few-Shot Learners", in Advances in Neural Information Processing Systems, 2020.

[24] L. Ouyang and others, "Training Language Models to Follow Instructions with Human Feedback", in Advances in Neural Information Processing Systems, 2022.

[25] OpenAI, "ChatGPT: Optimizing Language Models for Dialogue", 2022. Available: https://openai.com/blog/chatgpt

[26] S. Larson et al., "An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction", in Proceedings of EMNLP-IJCNLP, pp. 1311--1316, 2019. doi: 10.18653/v1/D19-1131.

[27] I. Casanueva et al., "Efficient Intent Detection with Dual Sentence Encoders", in Proceedings of the 2nd Workshop on NLP for Conversational AI, pp. 38--45, 2020. doi: 10.18653/v1/2020.nlp4convai-1.5.

[28] S. Gururangan et al., "Annotation Artifacts in Natural Language Inference Data", in Proceedings of NAACL-HLT, pp. 107--112, 2018. doi: 10.18653/v1/N18-2017.

[29] M. Geva, Y. Goldberg, and J. Berant, "Are We Modeling the Task or the Annotator? An Investigation of Annotator Bias in Natural Language Understanding Datasets", in Proceedings of EMNLP-IJCNLP, pp. 1161--1166, 2019. doi: 10.18653/v1/D19-1107.

[30] S. Swayamdipta et al., "Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics", in Proceedings of EMNLP, pp. 9275--9293, 2020. doi: 10.18653/v1/2020.emnlp-main.746.

[31] Y. Belinkov and Y. Bisk, "Synthetic and Natural Noise Both Break Neural Machine Translation", in Proceedings of ICLR, 2018.

[32] Z. Sun, A. Li, and X. Qiu, "Adversarial Training for Code-Mixed Language Understanding", in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 1364--1373, 2020. doi: 10.18653/v1/2020.acl-main.125.
