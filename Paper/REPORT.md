# College AI Assistant: A Multi-Signal, Domain-Constrained Conversational System

## Literature Survey
Conversational AI systems are commonly organized as (i) an NLU layer that maps user language to structured intent/state and (ii) a response layer that produces natural language outputs. The selected works below are consolidated and rewritten using the technically strongest explanations from the available draft surveys, while preserving a logical progression from traditional sequence models, to transformer-based representation learning, to dialogue alignment, and finally to intent evaluation with explicit out-of-scope (OOS) handling.

### Traditional Methods → Transformers → Dialogue Systems → Intent Evaluation

#### Hussain et al. (2019) — Conversational agent taxonomy and design techniques \cite{hussain2019survey}
- Problem: The chatbot design space lacks a consistent taxonomy, making it difficult to compare systems and reason about controllability versus coverage.
- Method: Systematic survey that classifies agents as rule-based, retrieval-based, or generative and summarizes common implementation patterns.
- Contribution: Establishes an organizing framework that motivates domain scoping and hybrid architectures for constrained deployments.
- Limitation: Characterization of generative systems largely predates large-scale transformer alignment methods, underrepresenting RLHF-era safety and preference optimization.

#### Caldarini et al. (2022) — Recent chatbot advances and evaluation practices \cite{caldarini2022literature}
- Problem: Rapid growth of chatbot research has produced fragmented evaluation practices and inconsistent reporting across systems and domains.
- Method: Systematic literature survey of modern chatbot architectures, datasets, and evaluation methods (2019–2021).
- Contribution: Provides consolidated trends (architecture families, application domains, evaluation protocols) and highlights the continued reliance on human evaluation.
- Limitation: As a survey, it does not provide deployable algorithms and cannot resolve limitations of the underlying benchmarks.

#### Liu and Lane (2016) — Joint intent classification and slot filling \cite{liu2016joint}
- Problem: Pipeline NLU architectures propagate errors between intent detection and slot filling.
- Method: Attention-based bidirectional RNN trained jointly for utterance-level intent classification and token-level slot tagging.
- Contribution: Demonstrates that shared representations improve joint performance compared to independent pipeline components.
- Limitation: Sequential recurrence reduces parallelism and long-context modeling; the formulation is utterance-centric rather than multi-turn.

#### Hakkani-Tur et al. (2016) — Multi-domain joint semantic frame parsing \cite{hakkani2016multidomain}
- Problem: Real assistants must support multiple domains while maintaining consistent semantic parsing quality.
- Method: BiRNN-LSTM model trained to jointly predict domain/intent/slots, enabling shared representations across domains.
- Contribution: Shows that multi-domain joint training can improve low-resource domain performance through transfer.
- Limitation: Fixed ontologies and domain boundaries remain brittle; scaling to many domains increases ambiguity and motivates explicit OOS handling.

#### Vaswani et al. (2017) — Transformer self-attention \cite{vaswani2017attention}
- Problem: RNN-based sequence-to-sequence models are inherently sequential and struggle with long-range dependencies.
- Method: Encoder–decoder transformer using multi-head scaled dot-product self-attention and positional encodings.
- Contribution: Replaces recurrence with attention, enabling parallel training and establishing the foundation for modern pre-trained models.
- Limitation: Self-attention has quadratic $O(n^2)$ cost in sequence length, limiting practical context windows without further architectural changes.

#### Devlin et al. (2019) — BERT bidirectional pre-training for NLU \cite{devlin2019bert}
- Problem: Unidirectional pre-training yields weaker contextual embeddings for classification and extraction tasks.
- Method: Bidirectional transformer encoder pre-trained with masked language modeling (and next sentence prediction), then fine-tuned.
- Contribution: Produces transfer-ready contextual representations that substantially improve intent classification and slot tagging without manual feature engineering.
- Limitation: Encoder-only architecture is not a response generator and can be overconfident without explicit calibration and OOS-aware routing.

#### Liu et al. (2019) — RoBERTa and robust pre-training recipes \cite{liu2019roberta}
- Problem: Downstream performance can be limited by suboptimal pre-training choices rather than model architecture.
- Method: Refines BERT-style training by removing next sentence prediction, training longer on more data with larger batches, and using dynamic masking.
- Contribution: Demonstrates that scaling data/compute and simplifying objectives yield consistent improvements, strengthening transformer encoders as NLU backbones.
- Limitation: Improved representations do not eliminate the need for domain scoping, OOS detection, and uncertainty-aware decision thresholds.

#### Radford et al. (2018, 2019) — GPT-style generative pre-training \cite{radford2018gpt,radford2019gpt2}
- Problem: Task-specific modeling does not scale across diverse language tasks, limiting reuse across conversational settings.
- Method: Autoregressive transformer decoder pre-training followed by task adaptation (fine-tuning or prompt conditioning).
- Contribution: Establishes generative pre-training as a practical response-layer foundation and motivates prompt-based adaptation.
- Limitation: Unconstrained generation can hallucinate domain facts, motivating guardrails, grounding, and scope control in deployed assistants.

#### Brown et al. (2020) — Few-shot learning with large language models \cite{brown2020fewshot}
- Problem: Fine-tuning for each new task or domain is expensive and can fail to generalize under distribution shift.
- Method: Scaling autoregressive transformers and evaluating in-context few-shot learning via prompt demonstrations.
- Contribution: Shows prompting as an adaptation interface, enabling rapid task conditioning without gradient updates.
- Limitation: High inference cost and limited grounding yield fluent but incorrect outputs, especially in constrained domains.

#### Khatri et al. (2018) — Alexa Prize: open-domain system lessons \cite{khatri2018alexa}
- Problem: Academic metrics underrepresent the system-integration challenges of sustained open-domain dialogue with real users.
- Method: Competition-driven deployment of socialbots combining topic tracking, retrieval, generation, and multi-skill orchestration.
- Contribution: Highlights that robust user experience depends on hybrid architectures, graceful failure handling, and long-run evaluation.
- Limitation: Results are difficult to reproduce outside the Alexa ecosystem, and user ratings are noisy and confounded by context.

#### Roller et al. (2021) — Recipes for building open-domain chatbots \cite{roller2021recipes}
- Problem: Purely generative dialogue models can be inconsistent and hard to control across topics and turns.
- Method: Empirical “recipes” blending retrieval augmentation, generation, and evaluation discipline (e.g., BlenderBot-style systems).
- Contribution: Motivates multi-signal orchestration that combines deterministic control with generation, emphasizing human evaluation.
- Limitation: Many practices assume large-scale data and annotation budgets that are not available in small projects.

#### Ouyang et al. (2022) — RLHF for instruction following (InstructGPT) \cite{ouyang2022rlhf}
- Problem: Next-token pre-training optimizes likelihood rather than helpfulness, honesty, and harmlessness.
- Method: Supervised fine-tuning on demonstrations, reward modeling from human preferences, and policy optimization (RLHF).
- Contribution: Improves instruction adherence and reduces undesirable outputs, forming a widely adopted alignment template.
- Limitation: Human feedback is costly and can encode annotator bias; reward models can be exploited by optimization artifacts.

#### Thoppilan et al. (2022) — Dialogue-optimized language models (LaMDA) \cite{thoppilan2022lamda}
- Problem: General-purpose PLMs do not directly optimize dialogue-specific qualities such as safety and groundedness.
- Method: Dialogue-focused training and tuning with objectives emphasizing quality, safety, and groundedness.
- Contribution: Demonstrates that dialogue-specialized optimization improves human-rated conversational behavior.
- Limitation: Resource requirements and residual factual errors motivate additional guardrails (scope control, grounding, and monitoring).

#### OpenAI (2022) — ChatGPT as a dialogue-aligned assistant \cite{openai2022chatgpt}
- Problem: Large language models require alignment to behave as usable assistants in interactive dialogue.
- Method: Dialogue-focused fine-tuning and preference optimization in the RLHF family.
- Contribution: Demonstrates that assistant behavior can be shaped toward instruction adherence and improved conversational usability.
- Limitation: Alignment improves helpfulness but does not guarantee factual grounding, particularly for restricted domains.

#### Chen et al. (2019) — BERT for joint intent classification and slot filling \cite{chen2019bertjoint}
- Problem: Traditional joint NLU models require task-specific feature engineering and struggle to exploit deep contextual cues.
- Method: Fine-tunes BERT for intent (via [CLS]) and slots (token-level tagging, commonly paired with a CRF).
- Contribution: Establishes a strong joint NLU baseline with substantial gains over BiLSTM-based models.
- Limitation: Inference cost can be high for latency-sensitive systems, and closed-set benchmarks may overestimate real-world robustness.

#### Xia et al. (2018) — Capsule networks for joint slot and intent \cite{xia2018capsule}
- Problem: Flat sequence encoders may fail to model hierarchical relationships between slot spans and intent labels.
- Method: Capsule networks with dynamic routing to represent part–whole relationships between tokens, slots, and intents.
- Contribution: Introduces a structured inductive bias that can improve robustness on ambiguous utterances.
- Limitation: Routing adds complexity and compute; benefits can be sensitive to dataset characteristics.

#### Larson et al. (2019) — CLINC150 with explicit OOS evaluation \cite{larson2019clinc150}
- Problem: Many intent benchmarks omit OOS as a primary requirement, despite its centrality in deployed assistants.
- Method: Intent dataset spanning many intents/domains and including an explicit OOS label for rejection evaluation.
- Contribution: Operationalizes OOS detection and motivates calibrated confidence thresholds for deployment.
- Limitation: OOS distributions vary by application and organization, requiring domain-specific validation beyond a single benchmark.

#### Casanueva et al. (2020) — Banking77 and efficient intent detection \cite{casanueva2020banking77}
- Problem: Large transformer classifiers can be too slow or costly for interactive intent routing.
- Method: Dual sentence encoders with similarity-based classification, evaluated on the Banking77 dataset.
- Contribution: Motivates embedding-based intent matching as an efficient alternative or complement to heavier encoders.
- Limitation: Domain specificity limits transfer, and single-turn intent evaluation omits multi-turn state tracking.

#### Zhang et al. (2020) — Few-shot intent detection via contrastive learning \cite{zhang2020fewshot}
- Problem: New intents emerge frequently, but labeled data for each new intent is scarce.
- Method: Contrastive pre-training to learn intent-discriminative representations, followed by few-shot fine-tuning.
- Contribution: Improves intent generalization under limited supervision, supporting modular onboarding of new intents.
- Limitation: Performance is sensitive to negative sampling and dataset construction; extreme low-shot settings remain challenging.

#### Kim et al. (2019) — Dialogue state tracking as reading comprehension \cite{kim2019dst}
- Problem: Dialogue state tracking must extract slot values over multi-turn history, including values not seen during training.
- Method: Reformulates DST as reading comprehension over dialogue history, extracting slot values by querying the context.
- Contribution: Provides an ontology-light approach that generalizes to open-vocabulary values better than closed-set classification.
- Limitation: Per-slot querying can increase inference cost, and implicit/unspoken values remain difficult to recover.

#### Huang et al. (2020) — TRADE: transferable dialogue state generator \cite{huang2020transferable}
- Problem: Multi-domain DST requires tracking many slots and unseen values without domain-specific parameterization.
- Method: State generator with a copy mechanism that generates slot values token-by-token from dialogue context.
- Contribution: Enables cross-domain transfer and handles unseen slot values by copying from context.
- Limitation: Autoregressive generation increases latency and can degrade on long contexts or long-tail value distributions.

#### Maroengsit et al. (2019) — Evaluation methods for chatbots \cite{maroengsit2019evaluation}
- Problem: Chatbot quality spans multiple dimensions that are poorly captured by single automatic metrics.
- Method: Survey of automatic metrics and human evaluation protocols across chatbot types.
- Contribution: Encourages multi-metric reporting and careful interpretation of automated scores.
- Limitation: Many automatic metrics correlate weakly with human judgments in open-domain dialogue.

#### Gururangan et al. (2018) — Annotation artifacts in NLU benchmarks \cite{gururangan2018annotation}
- Problem: Benchmark performance can be inflated when datasets contain spurious lexical artifacts that correlate with labels.
- Method: Artifact analysis showing that models can succeed by exploiting shallow correlations.
- Contribution: Motivates skepticism toward single-dataset accuracy claims and supports robustness-oriented evaluation.
- Limitation: Artifact detection identifies risk but does not automatically produce domain-specific mitigations.

#### Geva et al. (2019) — Annotator bias and shortcut learning \cite{geva2019shortcut}
- Problem: Models may learn annotator-specific patterns rather than the intended semantic task.
- Method: Empirical analysis isolating annotator signals and shortcut features in NLU datasets.
- Contribution: Highlights measurement error in reported performance and motivates diverse, well-controlled annotation protocols.
- Limitation: Bias diagnosis requires careful experimental design and can be difficult to integrate into small-project evaluations.

#### Swayamdipta et al. (2020) — Dataset cartography via training dynamics \cite{swayamdipta2020dataset}
- Problem: Aggregate accuracy obscures which examples are easy, ambiguous, or systematically hard for models.
- Method: Uses training dynamics (confidence and variability across epochs) to map instances into difficulty regimes.
- Contribution: Provides a practical lens for diagnosing dataset difficulty and targeting evaluation/augmentation.
- Limitation: Requires instrumented training runs and is sensitive to model/training configuration.

#### Belinkov and Bisk (2018) — Noise sensitivity in neural models \cite{belinkov2018synthetic}
- Problem: Neural sequence models degrade sharply under realistic input noise (typos, character-level perturbations).
- Method: Evaluates robustness under synthetic and natural noise and quantifies performance degradation.
- Contribution: Motivates explicit preprocessing and robustness testing for deployed NLU components.
- Limitation: Findings generalize broadly but do not prescribe a single mitigation strategy suitable for all domains.

#### Sun et al. (2020) — Adversarial training for code-mixed/noisy inputs \cite{sun2020adversarial}
- Problem: Language understanding in multilingual and code-mixed user inputs is brittle under distribution shift.
- Method: Adversarial training strategies to improve robustness to perturbations and mixing patterns.
- Contribution: Supports robustness-oriented evaluation and defensive training for real-world conversational inputs.
- Limitation: Robustness techniques can be data- and compute-intensive relative to small-project constraints.

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

### 2. Architecture Diagram Explanation (textual)
Conceptually, an architecture diagram for the system contains the following blocks and directed connections:
- **User Interface** → sends the utterance $u_t$ to an **Orchestration Layer**.
- **Orchestration Layer** → routes $u_t$ through a **Preprocessing** block (normalization) and then to the **NLU block**.
- **NLU block** → produces (i) an **intent distribution** over a closed set of intents and (ii) an **uncertainty/confidence** value; optional auxiliary signals (e.g., sentiment/emotion) are treated as conditioning variables rather than primary decision targets.
- A **Scope/OOS Gate** consumes the intent distribution and confidence and outputs one of: *in-scope accept*, *clarification required*, or *out-of-scope reject*.
- A **Dialogue Context Manager** maintains structured state across turns and feeds a summarized context $m_t$ forward.
- For in-scope queries, a **Knowledge Grounding** component selects relevant domain facts $k_t$; the **Response Policy** constructs a response plan conditioned on $(u_t,m_t,k_t)$.
- The **LLM Generator** produces surface text conditioned on the response plan; outputs pass through a **Post-processing** and **Safety/Formatting** step.
- A **Telemetry Store** receives logs from each stage to support monitoring and evaluation.

The separation between control decisions and generation aligns with deployment guidance emphasizing modular guardrails for conversational systems \cite{khatri2018alexa,roller2021recipes,ouyang2022rlhf}.

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
**Intent classification evaluation.** The evaluation uses the repository intent patterns in [data/intents.json](data/intents.json), which contains **650** preprocessed patterns across **18** intent labels. A stratified 80/20 split is applied (**520** train, **130** test) with a fixed random seed (42). The same preprocessing logic used by the implementation ([text_preprocessor.py](text_preprocessor.py)) is applied prior to training and evaluation.

Baselines are derived from components available in the repository:
- **Majority baseline:** always predicts the most frequent training intent.
- **Keyword rules:** assigns the intent with the highest keyword match count (using the `keywords` fields in the intent JSON).
- **TF-IDF (1-NN cosine):** nearest-neighbor matching in TF-IDF space.
- **TF-IDF + Logistic Regression:** multi-class linear classifier over TF-IDF uni/bi-grams (the repository's TF-IDF configuration).

**End-to-end system tests.** End-to-end results are taken from the saved report `TEST_RESULTS_20260405_153750.txt`, which executes 35 scenario tests spanning in-domain requests, hybrid queries, explicit OOS queries, and boundary inputs.

### 3. Performance Results (tables)
**Intent classification (holdout split on repository patterns).**

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Majority baseline | 0.146 | 0.008 | 0.056 | 0.014 |
| Keyword rules | 0.477 | 0.647 | 0.414 | 0.456 |
| TF-IDF (1-NN cosine) | 0.654 | 0.727 | 0.543 | 0.567 |
| TF-IDF + Logistic Regression | 0.662 | 0.629 | 0.491 | 0.501 |

**Observations.**
- Keyword rules improve substantially over the majority baseline, indicating that intent-specific lexical cues exist in the pattern set.
- TF-IDF-based methods provide the strongest performance among fast lexical baselines; 1-NN has the highest macro F1 in this split, while logistic regression achieves slightly higher accuracy.
- These numbers are best interpreted as **pattern-set generalization** rather than a full measure of real-user robustness; pattern corpora can contain artifacts that inflate apparent performance \cite{gururangan2018annotation,geva2019shortcut}.

**End-to-end automated test outcomes (saved execution artifact).**

| Metric | Value |
|---|---:|
| Total tests | 35 |
| Passed | 33 (94.3%) |
| Failed | 2 (5.7%) |
| OOS detected | 6 |
| Latency (median / avg / max) | 2046.8 ms / 4370.6 ms / 13011.8 ms |

Category-level pass counts in the same artifact indicate that failures are isolated to boundary inputs (whitespace-only and punctuation-only queries), while in-domain and explicit OOS categories pass.

### 4. Comparison with Baseline Methods (if available)
The component-level comparison suggests that purely rule-based routing is competitive in precision but limited in recall (macro recall 0.414), reflecting reduced coverage for paraphrases and short queries. TF-IDF baselines improve recall and macro F1, supporting the use of statistical models for intent routing.

The project implementation additionally includes a semantic similarity intent component (Sentence-Transformers) and an ensemble strategy, motivated by benchmark findings that OOS-aware routing and robust intent representations are critical for deployed assistants \cite{larson2019clinc150,casanueva2020banking77}. Direct semantic-component metrics are not reported here because the reproducible holdout experiment above uses only the locally fit lexical baselines; however, the end-to-end test suite reflects the integrated system behavior.

### 5. Error Analysis
**Intent misclassification patterns (TF-IDF + Logistic Regression).** On the 130-example holdout test split, the best-accuracy lexical model produces 44 misclassifications. Representative errors reveal systematic ambiguity and overlap between intents:
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

**Key findings.** Quantitative results on repository artifacts indicate: (i) fast lexical baselines achieve up to 0.662 accuracy on a held-out split of intent patterns, with TF-IDF nearest-neighbor providing the strongest macro F1 (0.567) among the reported baselines; and (ii) end-to-end scenario tests achieve a 94.3\% pass rate (33/35), with failures concentrated on degenerate boundary inputs and a median latency of 2046.8 ms in the saved test artifact.

**Contributions.**
- A modular, repository-grounded architecture for a college assistant integrating intent routing, context management, scope gating, and an LLM layer.
- A practical evaluation setup that reports component-level intent metrics and end-to-end scenario outcomes from saved execution artifacts.
- Operational safeguards (confidence-aware routing, conservative fallback behavior, and error recovery) aligned with deployment reliability goals.

**Limitations.**
- Intent evaluation is based on locally authored patterns rather than large-scale external benchmarks, and may overestimate generalization due to dataset artifacts \cite{gururangan2018annotation,geva2019shortcut}.
- The semantic intent component and ensemble behavior are assessed indirectly through end-to-end tests rather than with separate reproducible component metrics.
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
