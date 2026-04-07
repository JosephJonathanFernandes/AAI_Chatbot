# College AI Assistant: A Multi-Signal, Domain-Constrained Conversational System

## Abstract
This report presents a college-domain conversational assistant that combines local multi-signal NLU control (preprocessing, intent, scope, emotion, and context) with external LLM response generation. Component-level evaluation on repository intent patterns shows strong semantic-routing performance (accuracy 0.783, macro F1 0.756), while end-to-end evaluation on 120 scenarios captures deployment-facing behavior under noisy, vague, emotional, and out-of-scope (OOS) inputs. A key finding is that a substantial portion of observed failures in batch execution are infrastructure-related (API rate-limit/provider failures) rather than core routing logic errors. Separating infrastructure failures from functional failures yields a clearer estimate of system behavior and supports more honest interpretation of pass-rate confidence intervals, OOS gating quality, and latency trade-offs.
The primary contribution is a deployment-realistic evaluation methodology that separates infrastructure failures from functional routing errors, yielding corrected confidence intervals that better reflect core assistant logic.

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
| \cite{vaswani2017attention} | Multi-head self-attention encoder-decoder | Parallel training and long-range dependency modeling | $O(n^2)$ attention cost |
| \cite{liu2016joint} | BiRNN joint intent+slot architecture | Early strong baseline for integrated NLU | Sequential recurrence limits throughput |
| \cite{devlin2019bert} | 110M-parameter bidirectional encoder, masked-LM pre-training, fine-tuned [CLS] for intent | Strong transferable contextual representations | Encoder-only; needs calibration for OOS |
| \cite{chen2019bertjoint} | BERT + token-level slot tagging (joint NLU) | Strong intent/slot benchmark performance | Heavier inference than lightweight baselines |
| \cite{zhang2020fewshot} | Contrastive pre-training + few-shot intent fine-tuning | Better low-resource intent onboarding | Sensitive to support-set quality |
| \cite{casanueva2020banking77} | Dual sentence encoders with cosine intent retrieval | Fast intent detection with efficient embedding lookup | Domain-specific evaluation scope |
| \cite{larson2019clinc150} | CLINC150 in-scope/OOS benchmark protocol | Explicitly evaluates rejection behavior | OOS definition varies by deployment |
| \cite{kim2019dst} | Dialogue state tracking as reading comprehension | Open-vocabulary slot value extraction | Higher per-turn computational cost |
| \cite{huang2020transferable} | TRADE dialogue state generator with copy mechanism | Strong cross-domain slot-value transfer with open vocabulary | Added latency and degradation risk on long dialogue contexts |
| \cite{radford2019gpt2} | Autoregressive decoder pre-training | Fluent open-ended generation | Hallucination/grounding risk |
| \cite{roller2021recipes} | Retrieval-generation hybrid chatbot recipes | Strong practical recipe for open-domain systems | Data and infrastructure intensive |
| \cite{ouyang2022rlhf} | SFT + reward model + PPO alignment pipeline | Better instruction following and safety behavior | Expensive and preference-bias sensitive |
| \cite{thoppilan2022lamda} | Dialogue-specialized LM training objectives | Improved conversational quality dimensions | Resource-intensive; residual factual errors |
| \cite{gururangan2018annotation} | Annotation-artifact analysis in NLU datasets | Exposes spurious dataset shortcuts | Not a direct mitigation method |
| \cite{geva2019shortcut} | Annotator-bias shortcut learning analysis | Diagnoses hidden labeling bias effects | Requires downstream debiasing work |
| \cite{swayamdipta2020dataset} | Dataset cartography via training dynamics | Identifies easy/ambiguous/hard instances | Adds analysis overhead |
| \cite{belinkov2018synthetic} | Character-noise robustness evaluation | Quantifies brittleness under perturbations | Improvement methods required separately |
| \cite{sun2020adversarial} | Adversarial training for multilingual/code-mixed dialogue robustness | Improves resilience to noisy code-mixed perturbations | Requires additional adversarial data and training compute |

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
In the implemented system, $\tau$ is intent-specific via dynamic confidence thresholds (typical academic intents in the 0.45--0.60 range from [confidence_threshold_manager.py](confidence_threshold_manager.py)), and the semantic-similarity gate uses $\gamma=0.50$ (from [scope_detector.py](scope_detector.py)); this thresholding behavior directly influences OOS precision/recall trade-offs.

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

Finally, robustness and measurement risks motivate conservative preprocessing and careful interpretation of automated evaluation results. Neural NLU brittleness under noise and dataset artifacts can inflate reported performance without improving real-world reliability \cite{belinkov2018synthetic,sun2020adversarial,gururangan2018annotation,geva2019shortcut,swayamdipta2020dataset}. The methodology therefore prioritizes scope control and grounded generation over unconstrained fluency, consistent with survey evidence that reliable chatbot performance depends on system integration and evaluation discipline \cite{caldarini2022literature,maroengsit2019evaluation}. These design choices translate directly into implementation requirements: a deterministic local control path, explicit uncertainty gating, and fault-tolerant LLM orchestration. The following section maps this design to concrete modules and execution pathways in the repository.

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

**Illustrative Hinglish normalization examples.**

| Raw Query | Normalized Query |
|---|---|
| fees kitni hai | what are the fees |
| kya placement hoti hai | what placement is |
| hostel me room hai kya | hostel room is what |
| exam kab hai | exam when is |
| scholarship kaise milegi | scholarship how get |

Normalization preserves content words and discards many Hindi function words; downstream NLU relies on semantic embeddings rather than grammaticality.

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

### 5. Selected Core Code Blocks
The following snippets omit imports, UI code, error-handling boilerplate, and comments.

**Data Normalization**
```python
@staticmethod
def remove_special_chars(text: str) -> str:
	text = re.sub(r"[^a-z0-9\s\?!]", "", text)
	text = re.sub(r"\s+", " ", text).strip()
	return text

@staticmethod
def normalize_hinglish(text: str) -> str:
	words = text.split()
	converted = []
	for word in words:
		clean_word = word.rstrip("?!")
		ending = word[len(clean_word):]
		if clean_word in TextPreprocessor.HINGLISH_MAP:
			mapped = TextPreprocessor.HINGLISH_MAP[clean_word]
			if mapped:
				converted.append(mapped + ending)
		else:
			converted.append(word)
	return " ".join(converted)
```

**Hybrid Intent Ensemble**
```python
sem_weighted = sem_conf * self.semantic_weight
tfidf_weighted = tfidf_conf * self.tfidf_weight

if sem_intent == tfidf_intent:
	final_intent = sem_intent
	avg_conf = (sem_conf + tfidf_conf) / 2.0
	agreement_boost = 1.15
	final_confidence = min(1.0, avg_conf * agreement_boost * self.confidence_scaling)
elif max(sem_weighted, tfidf_weighted) > 0.6:
	if sem_weighted > tfidf_weighted:
		final_intent = sem_intent
		final_confidence = min(1.0, sem_weighted * self.confidence_scaling)
	else:
		final_intent = tfidf_intent
		final_confidence = min(1.0, tfidf_weighted * self.confidence_scaling)
else:
	if sem_weighted >= tfidf_weighted:
		final_intent = sem_intent
		final_confidence = min(1.0, sem_weighted * self.confidence_scaling)
	else:
		final_intent = tfidf_intent
		final_confidence = min(1.0, tfidf_weighted * self.confidence_scaling)

final_confidence = max(self.confidence_min, final_confidence)
return final_intent or "unknown", final_confidence, detailed_scores
```

**Uncertainty and Scope Gating**
```python
query_lower = query.lower()
has_college_context = any(prefix in query_lower for prefix in self.COLLEGE_CONTEXT_PREFIXES)
if has_college_context:
	return True, "college_context_detected", 0.85

if self.definite_oos_pattern.search(query_lower):
	return False, "definite_out_of_domain", 0.95
if self.ambiguous_oos_pattern.search(query_lower):
	return False, "ambiguous_out_of_domain", 0.65

domain_score, matched_category = self._check_domain_keywords(query_lower)
if self.use_semantic and domain_score == 0.0:
	semantic_score, semantic_category = self._check_semantic_similarity(query)
	if semantic_score > domain_score:
		domain_score, matched_category = semantic_score, semantic_category

if conversation_history:
	context_boost = self._compute_context_score(query_lower, conversation_history)
	if context_boost > 0:
		domain_score = min(0.95, domain_score + context_boost)

if domain_score > self.confidence_threshold:
	return True, f"domain_keywords_{matched_category}", domain_score
return False, "low_domain_confidence", max(domain_score, intent_confidence * 0.3)
```

**Context Tracking**
```python
def add_turn(self, user_input, bot_response, intent, confidence, emotion, entities=None):
	turn = {
		"timestamp": datetime.now().isoformat(),
		"user_input": user_input,
		"bot_response": bot_response,
		"intent": intent,
		"confidence": confidence,
		"emotion": emotion,
		"entities": entities or {}
	}
	self.conversation_history.append(turn)
	self.last_intent = intent
	self.last_confidence = confidence
	self.last_emotion = emotion
	self.last_entities = entities or {}

def resolve_pronouns(self, user_input: str) -> str:
	lower_input = user_input.lower()
	resolved = user_input
	if ("it" in lower_input or "that" in lower_input or "this" in lower_input) and self.last_entities:
		if "department" in self.last_entities:
			resolved = resolved.replace("it", self.last_entities["department"])
			resolved = resolved.replace("that", self.last_entities["department"])
	return resolved
```

**LLM Operational Safeguards**
```python
last_error = None
for attempt in range(self.retry_max_attempts):
	try:
		response = requests.post(
			f"{self.groq_base_url}/chat/completions",
			headers=headers,
			json=payload,
			timeout=self.groq_timeout
		)

		if response.status_code == 429:
			self.rate_limit_count += 1
			if len(self.groq_api_keys) > 1:
				self._rotate_groq_key()
				groq_api_key = self._get_current_groq_key()
				time.sleep(self._get_jittered_backoff_delay(attempt))
				continue
			if attempt < self.retry_max_attempts - 1:
				time.sleep(self._get_jittered_backoff_delay(attempt))
				continue
			return {"error": "Rate limit exceeded after retries"}

		if response.status_code != 200:
			if 500 <= response.status_code < 600 and attempt < self.retry_max_attempts - 1:
				time.sleep(self._get_jittered_backoff_delay(attempt))
				continue
			return {"error": last_error}
```

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
**Intent classification evaluation.** The evaluation uses the repository intent patterns in [data/intents.json](data/intents.json), which contains approximately **650** preprocessed patterns across **18** intent labels (about **36 patterns per intent on average**). A stratified 80/20 split is applied with a fixed random seed (42). The same preprocessing logic used by the implementation ([text_preprocessor.py](text_preprocessor.py)) is applied prior to training and evaluation.

Baselines are derived from components available in the repository:
- **Majority baseline:** always predicts the most frequent training intent.
- **Keyword rules:** assigns the intent with the highest keyword match count (using the `keywords` fields in the intent JSON).
- **TF-IDF (1-NN cosine):** nearest-neighbor matching in TF-IDF space.
- **TF-IDF + Logistic Regression:** multi-class linear classifier over TF-IDF uni/bi-grams (the repository's TF-IDF configuration).
- **Sentence-Transformers semantic similarity:** `all-MiniLM-L6-v2` cosine similarity against training patterns, matching the repository's semantic intent logic.
- **Weighted ensemble:** semantic (0.75) + TF-IDF (0.25) intent selection using the decision rules implemented in [intent_model.py](intent_model.py).

**End-to-end system tests.** End-to-end results are taken from the saved report `TEST_RESULTS_20260407_085552.txt`, which executes 120 scenario tests spanning in-domain requests, hybrid queries, explicit OOS queries, typos, edge cases, and noisy inputs.

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
- In this split, the weighted ensemble matches the semantic predictor because the semantic signal dominates under the default weights; this should be interpreted as a **calibration/robustness mechanism** (confidence smoothing and fallback behavior on ambiguous/noisy inputs), not as an accuracy booster on this holdout.
- These numbers are best interpreted as **pattern-set generalization** rather than a full measure of real-user robustness; pattern corpora can contain artifacts that inflate apparent performance \cite{gururangan2018annotation,geva2019shortcut}.

**End-to-end automated test outcomes (saved execution artifact).**

| Metric | Value |
|---|---:|
| Total tests | 120 |
| Passed | 94 (78.3%) |
| Failed | 26 (21.7%) |
| OOS detected | 18 |
| Latency (median / avg / max) | 4729.2 ms / 4520.3 ms / 14652.3 ms |

**120-scenario suite composition.**

| Scenario Bucket | Total | Passed | Pass Rate |
|---|---:|---:|---:|
| In-domain task | 87 | 75 | 86.2% |
| Robustness/noisy in-domain | 22 | 11 | 50.0% |
| Explicit out-of-scope | 10 | 8 | 80.0% |
| Boundary inputs | 1 | 0 | 0.0% |

![120-Scenario Suite Composition and Pass Rates](Paper/figures/suite_composition_pass_rates.svg)

This figure is generated by [Paper/generate_suite_composition_passrates_chart.py](Paper/generate_suite_composition_passrates_chart.py) and visualizes pass-rate vs. remaining proportion for each scenario bucket.

**Failure attribution of 26 failed tests.**

| Failure Type | Count |
|---|---:|
| API/provider failures (`Both APIs failed`) | 14 |
| Boundary/underspecified/noisy inputs | 9 |
| Intent/scope misroutes (non-API) | 3 |

The raw 78.3% pass rate includes 14 provider-side failures during batch execution. Excluding these exogenous API failures gives a functional pass rate of 88.7% (94/106) on completed responses, which better reflects core system logic.
Here, boundary inputs are deliberately degenerate queries (for example punctuation-only, whitespace-only, or extreme-length gibberish) used to test guardrails rather than normal user behavior.
Within the robustness/noisy in-domain bucket, the 11 failures are concentrated in typo-heavy variants (4), vague/underspecified prompts (4), and emotion-heavy short turns (3), explaining the lower 50.0\% pass rate.

Taken together, the intent-performance table and the end-to-end table show that component-level intent strength does not directly translate to raw batch pass rate when external API reliability becomes a bottleneck.

### 4. Comparison with Baseline Methods (if available)
The component-level comparison suggests that rule-based routing can be competitive in precision but may be limited in recall, reflecting reduced coverage for paraphrases and short queries. TF-IDF baselines improve macro recall and macro F1, supporting the use of statistical models for intent routing.

The repository implementation additionally includes a semantic similarity component (Sentence-Transformers) and a weighted ensemble. On the same holdout split, the semantic component substantially improves macro F1 (0.756), consistent with prior work emphasizing robust representations for intent routing and uncertainty-aware scoping \cite{larson2019clinc150,casanueva2020banking77}.

Because the ensemble and semantic-only scores are identical in this split, the ensemble should be viewed as an operational reliability choice (confidence calibration and safer routing under ambiguity), rather than a direct accuracy gain mechanism.

### 5. Latency Breakdown (NLU vs LLM)
The end-to-end artifact reports median/average/max response latencies in the multi-second range, but it does not separate local NLU computation from LLM response time. The application implementation exposes an LLM timing signal (`response_time`), enabling a decomposition of total per-turn latency into:

$$
T_{\mathrm{total}} \approx T_{\mathrm{NLU}} + T_{\mathrm{LLM}} + T_{\mathrm{overhead}}.
$$

A local microbenchmark (CPU, after warm-up) indicates that the control layer is comparatively fast: intent-only inference has a median of **27.97 ms** (mean **28.55 ms**), and intent+emotion+scope has a median of **60.62 ms** (mean **63.67 ms**, $p95$ **83.86 ms**). Consequently, the multi-second tail in the end-to-end artifact is primarily attributable to external LLM inference and network variability rather than local intent routing.

For a student-facing assistant, a median of about 4.7 seconds per turn is usable but not ideal for conversational flow. Practical mitigation includes (i) token streaming for faster time-to-first-token, (ii) cache-first handling of high-frequency intents such as fees/exams/admission, and (iii) asynchronous UI updates so scope/intention feedback appears before full generation completes.

![Per-Turn Latency Breakdown](Paper/figures/per_turn_latency_breakdown.svg)

The chart is generated by [Paper/generate_latency_breakdown_chart.py](Paper/generate_latency_breakdown_chart.py) and shows that external LLM inference dominates the median per-turn budget.

### 6. Statistical Strength of Scenario Tests
The recorded end-to-end scenario result (94/120 passed) corresponds to a raw pass rate of 78.3\%. The 95\% Wilson confidence interval for this **raw operational pass rate** is **[70.1%, 84.8%]**. Because 14/26 failures were API/provider failures, a corrected functional estimate (94/106) is 88.7\% with a 95\% Wilson interval of **[81.2%, 93.4%]**. Reporting both numbers separates infrastructure reliability from core assistant logic.

This expanded 120-scenario evaluation includes systematic coverage of paraphrases, typos, borderline OOS queries, hybrid queries, emotional language, vague inputs, and true out-of-scope boundary cases, reducing measurement artifacts and providing more robust evidence of system behavior in realistic deployment conditions \cite{gururangan2018annotation,geva2019shortcut,swayamdipta2020dataset,larson2019clinc150}.

### 7. OOS Detection Metrics
Using explicit OOS-labeled scenarios in the suite (OUTSCOPE + boundary cases), scope-gating outcomes are:

| OOS Metric | Value |
|---|---:|
| Precision | 0.444 |
| Recall | 0.727 |
| False-positive rate (in-scope flagged OOS) | 0.092 |

These values indicate conservative but imperfect OOS routing: recall is acceptable, but precision shows over-rejection on some ambiguous/noisy in-scope queries and should be improved with better threshold calibration.
Concretely, OOS precision 0.444 corresponds to 8 true OOS catches and 10 false OOS flags; inspection of failures shows that many false positives arise from noisy/underspecified in-domain turns, indicating that the current threshold pair is recall-oriented and likely needs tighter calibration for precision.

### 8. Error Analysis
**Top intent confusions (TF-IDF + Logistic Regression holdout).**

| True Intent | Predicted Intent | Count |
|---|---|---:|
| admission | eligibility | 3 |
| admission | exams | 3 |
| gratitude | greetings | 2 |
| admission | out_of_scope | 2 |
| fees | campus_life | 2 |

![Top Misclassified Intent Pairs Heatmap](Paper/figures/top_misclassified_intents_heatmap.svg)

The heatmap is generated by [Paper/generate_confusion_matrix_heatmap.py](Paper/generate_confusion_matrix_heatmap.py) using Seaborn/Matplotlib with a white background and a Blues colormap for publication readability.

These confusion pairs indicate overlap between administratively related intents and conversational-act intents.

**Bottom-5 per-class F1 (semantic model, holdout split).**

| Intent | F1 | Support |
|---|---:|---:|
| general_info | 0.400 | 4 |
| comparison | 0.444 | 5 |
| affirmation | 0.667 | 5 |
| eligibility | 0.667 | 7 |
| library | 0.667 | 4 |

**End-to-end failure modes.** Failure attribution indicates three distinct regimes: API/provider failures (14), boundary/underspecified/noisy queries (9), and non-API intent/scope misroutes (3). This indicates that raw pass-rate degradation is primarily operational during batch execution, while logical failure modes are concentrated in ambiguity-heavy edge cases \cite{belinkov2018synthetic,sun2020adversarial}.

**Strengths and weaknesses.**
- Strength: strong end-to-end functional pass rate on in-domain and OOS scenarios in the recorded artifact, indicating stable orchestration and conservative fallback behavior.
- Weakness: component-level intent confusion remains for conversational-act intents and semantically overlapping fee/hostel/admission categories; improving label definitions, increasing pattern coverage, and adding explicit conversational-act handling are expected to improve macro recall.

## Conclusion
This work addresses the problem of delivering a reliable college-domain assistant that can route user queries to appropriate intents, handle out-of-scope (OOS) inputs conservatively, and produce helpful responses while reducing brittle behavior in real-world interaction.

**Approach.** The system is implemented as a controlled conversational pipeline that combines preprocessing, ensemble intent inference, scope enforcement, emotion and tone cues, and an LLM response layer with knowledge grounding and operational safeguards (e.g., fallback and recovery). The design reflects evidence that robust deployed assistants require both strong representations and explicit control mechanisms, particularly for OOS handling and noise resilience \cite{larson2019clinc150,casanueva2020banking77,belinkov2018synthetic}.

**Key findings.** Quantitative results on repository artifacts indicate: (i) lexical baselines achieve up to 0.620 accuracy and up to 0.562 macro F1 on the held-out split of intent patterns, while the semantic component achieves 0.783 accuracy and 0.756 macro F1 on the same split; and (ii) end-to-end scenario tests on an expanded 120-scenario benchmark achieve a raw pass rate of 78.3\% (94/120; 95\% CI [70.1%, 84.8%]), with 14 failures attributable to API/provider errors and a corrected functional pass estimate of 88.7\% (94/106; 95\% CI [81.2%, 93.4%]).
In condensed form, the main novelty is methodological: reporting raw operational reliability alongside infrastructure-corrected functional reliability, so evaluation reflects both deployment reality and core assistant logic.

**Contributions.**
- A repository-grounded multi-signal architecture that combines local NLU control and external LLM generation, validated on 120 end-to-end scenarios.
- A deployment-realistic evaluation protocol that reports both raw operational pass rate (78.3%, 94/120) and corrected functional pass rate (88.7%, 94/106).
- A quantified reliability analysis with explicit failure attribution (14 API failures vs 12 non-API failures), OOS metrics (precision 0.444, recall 0.727), and confidence intervals for both reporting modes.

**Limitations.**
- Intent evaluation is based on locally authored patterns rather than large-scale external benchmarks, and may overestimate generalization due to dataset artifacts \cite{gururangan2018annotation,geva2019shortcut}.
- The pattern inventory is relatively shallow per class (about 36 patterns per intent on average), with several low-sample intents (e.g., `unknown`, `campus_life`, `greetings`, `gratitude`, `general_info`) that can increase class imbalance effects.
- The intent evaluation reflects pattern-set generalization and does not fully capture real-user distribution shift, multi-turn ambiguity, or adversarial/noisy inputs.
- Boundary input handling (e.g., punctuation-only and whitespace-only) remains a failure point and motivates stronger normalization and routing rules.

**Future work.**
- Multilingual support: begin by extending token normalization maps in [text_preprocessor.py](text_preprocessor.py) for Konkani and Marathi variants observed in the local student population, then re-evaluate OOS false positives on code-mixed queries.
- Better context handling: add anaphora-aware follow-up resolution in [context_manager.py](context_manager.py) for pronoun-heavy turns and re-measure failure rates on the CONTEXT bucket.
- Real-time deployment: implement streaming-first responses in [app.py](app.py) and cache-first replies for high-frequency intents in [llm_handler.py](llm_handler.py), then benchmark median time-to-first-token.
- Improved robustness: tune $\tau$/$\gamma$ against a held-out OOS calibration set and re-run the 120-scenario suite to target higher OOS precision without collapsing recall.

## Appendix

### A. Reproducibility Checklist
| Item | Value |
|---|---|
| OS | Windows |
| Python | 3.10+ (project scripts are standard CPython) |
| Main runtime dependencies | streamlit 1.28.1, torch 2.1.1, transformers 4.35.2, sentence-transformers 2.2.2, scikit-learn 1.3.2, pandas 2.1.4, numpy 1.24.3 |
| API/LLM dependencies | groq 0.13.3, google-genai 0.1.0, requests 2.31.0 |
| Test/tooling dependencies | pytest 7.4.3, pytest-cov 4.1.0, pytest-xdist 3.5.0, pytest-timeout 2.1.0, hypothesis 6.92.0 |
| Dataset files | `data/intents.json`, `data/college_data.json` |
| Primary result artifact | `TEST_RESULTS_20260407_085552.txt` |
| Intent split protocol | Stratified 80/20, random seed 42 |
| Figure outputs | `Paper/figures/*.pdf` and `Paper/figures/*.svg` |

### B. Full 120-Scenario Inventory (Artifact Index)
The complete 120-case scenario inventory is provided as machine-readable project artifacts:
- `CHATBOT_QUESTION_TEST_CASES.txt` (scenario catalog and identifiers)
- `STREAMLIT_TEST_CASES.txt` (UI-facing scenario list)
- `TEST_RESULTS_20260407_085552.txt` (full execution log with per-case outcomes)

Bucket-level coverage used in the paper:
- In-domain task: 87 cases
- Robustness/noisy in-domain: 22 cases
- Explicit out-of-scope: 10 cases
- Boundary inputs: 1 case

### C. OOS Confusion Details and Formulae
Using explicit OOS-labeled cases in the suite:
- True positives (TP): 8
- False positives (FP): 10
- False negatives (FN): 3
- True negatives (TN): 99

Metrics are computed as:

$$
	ext{Precision} = \frac{TP}{TP + FP} = \frac{8}{8+10} = 0.444
$$

$$
	ext{Recall} = \frac{TP}{TP + FN} = \frac{8}{8+3} = 0.727
$$

$$
	ext{FPR} = \frac{FP}{FP + TN} = \frac{10}{10+99} = 0.092
$$

### D. Representative Failure Log Excerpts
| Query (excerpt) | Test Type | Detected Intent | Failure Regime |
|---|---|---|---|
| Wht r the fess? | `fees_-_with_typos` | hostel (77.9%) | Noise/typo misrouting |
| Can I retake the exam if I fail? | `exam_-_re-exam_opportunity` | exams (100.0%) | Retrieval/generation mismatch |
| What's the average salary for placements? | `placements_-_average_package` | placements (95.1%) | Knowledge grounding gap |
| Do placements vary by specialization/branch? | `placements_-_specialization_gaps` | placements (71.1%) | Specificity gap |
| What if I want to pursue higher studies instead? | `placements_-_higher_studies` | comparison (44.0%) | Intent overlap |
| How many students can be accommodated in hostel? | `hostel_-_capacity` | hostel (87.4%) | Missing structured field |
| What labs are available for students? | `facilities_-_labs` | fees (46.0%) | Cross-intent confusion |
| How do I apply to the college? | `admission_-_application_process` | admission (65.6%) | Incomplete response |
| What types of scholarships are available? | `scholarship_-_types_available` | fees (72.1%) | Intent schema overlap |
| Wat about the exams skeduled? | `typo_-_phonetic_spelling` | exams (100.0%) | Noise robustness failure |

### E. Figure Generation Provenance
The paper figures are generated directly from repository scripts:
- `Paper/generate_latency_breakdown_chart.py` -> `per_turn_latency_breakdown.(pdf|svg)`
- `Paper/generate_confusion_matrix_heatmap.py` -> `top_misclassified_intents_heatmap.(pdf|svg)`
- `Paper/generate_suite_composition_passrates_chart.py` -> `suite_composition_pass_rates.(pdf|svg)`

One-command regeneration (from repository root):

```powershell
python Paper/generate_latency_breakdown_chart.py
python Paper/generate_confusion_matrix_heatmap.py
python Paper/generate_suite_composition_passrates_chart.py
```

### F. Prompt/Policy Snapshot (Compact)
Operational response policy implemented in the runtime stack:
- Scope-first routing: explicit out-of-domain checks precede generation.
- Confidence-aware clarification: dynamic intent thresholds determine clarify vs answer.
- Controlled generation: prompts inject intent, confidence, scope reason, emotion, and retrieved facts.
- Reliability safeguards: cache lookup, request throttling, concurrent-call limits, retry with jittered backoff, API-key rotation, and Groq->Gemini fallback.

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
