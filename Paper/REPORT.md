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

## Methodology (Design)
The implemented assistant follows a multi-signal architecture that combines deterministic control components with a generative response layer. The design reflects common findings in conversational AI surveys: strong user experience depends on combining NLU, state/context tracking, and robust evaluation protocols rather than relying solely on surface-level fluency \cite{caldarini2022literature,gao2020recent,gao2019neural,maroengsit2019evaluation}.

Design objectives:
- Domain constraint: Enforce a clear in-scope boundary for a college-information domain.
- Multi-signal control: Use intent, emotion, and scope signals to modulate response behavior.
- Grounded generation: Prefer responses grounded in a local knowledge base when possible.
- Operational robustness: Apply throttling, retries, caching, and fallback strategies for LLM calls.

The design also treats uncertainty as a first-class signal: confidence-based intent routing and explicit out-of-scope handling are aligned with benchmarks that stress OOS rejection \cite{larson2019clinc150,casanueva2020banking77}. Evaluation claims are constrained by known risks of dataset artifacts and bias in automated measurements \cite{gururangan2018annotation,geva2019shortcut,swayamdipta2020dataset}.

## Implementation
The implementation is grounded in the repository modules and follows a fixed processing pipeline designed to minimize hallucination risk and reduce brittle behavior under noisy inputs.

Key components:
- UI: Streamlit orchestration in `app.py` initializes cached model instances and executes a deterministic per-message pipeline.
- Intent: An ensemble combines embedding-based semantic similarity with TF–IDF + logistic regression voting and confidence calibration (see `intent_model.py`).
- Emotion: A cached transformer sentiment pipeline and rule-based refinements provide an emotion signal (`emotion_detector.py`, `emotional_tone_detector.py`).
- Scope: Rule/keyword scoring with optional semantic similarity determines in-scope vs out-of-scope queries (`scope_detector.py`).
- Context and time: Lightweight conversation context management and time-aware features support multi-turn coherence and greetings (`context_manager.py`, `time_context.py`, `session_greeter.py`).
- Threshold management: Centralized confidence-threshold logic is used to standardize acceptance, clarification, and OOS behavior (`confidence_threshold_manager.py`).
- LLM: A prompt-engineered Groq primary path with Gemini fallback, concurrency limiting, retries/backoff/jitter, key rotation, and response caching (`llm_handler.py`, `prompt_engineering.py`).
- Persistence: SQLite logs telemetry (intent/confidence, emotion, scope, latency, LLM source) (`database.py`).
- Resilience: Error recovery utilities help isolate failures and maintain a stable user experience under external API and parsing errors (`error_recovery.py`).

Processing flow (conceptual):
- Text normalization and lightweight preprocessing occur before classification to reduce sensitivity to formatting artifacts and punctuation-only inputs, which are common sources of NLU brittleness \cite{sun2020adversarial,belinkov2018synthetic}.
- Intent inference uses a hybrid of sparse lexical evidence (TF-IDF + logistic regression) and dense semantic evidence (embedding similarity), reflecting a practical trade-off between interpretability, latency, and generalization across paraphrases \cite{devlin2019bert,liu2019roberta,casanueva2020banking77}.
- When confidence is insufficient, the architecture is designed to prefer conservative behaviors (e.g., clarify, route to a safer template, or enforce out-of-scope handling) rather than forcing a brittle intent decision \cite{larson2019clinc150}.
- Response generation is separated from control: the LLM is invoked after scope/intent decisions are computed, aligning with system-level guidance that emphasizes modular guardrails and evaluation discipline \cite{hussain2019survey,khatri2018alexa,roller2021recipes}.
- Context and temporal cues are applied consistently at the orchestration layer so that greetings and follow-ups do not bypass scope and confidence checks.

## Results
A saved automated test execution artifact reports 35 tests with 33 passes (94.3%) and two failures on boundary inputs (whitespace-only and punctuation-only queries). Reported latency statistics include a maximum of 13011.8 ms and a median of 2046.8 ms (see `TEST_RESULTS_20260405_153750.txt`).

Interpretation constraint: The same artifact frequently records `Intent: None (0.0%)` and a default emotion label across semantically meaningful inputs, indicating a likely mismatch between the test harness interface and the current classifier return contracts. Consequently, the aggregate pass rate is best interpreted as end-to-end response generation success under that harness rather than a validated measure of intent/emotion correctness \cite{maroengsit2019evaluation,gururangan2018annotation,geva2019shortcut,swayamdipta2020dataset}.

Deployment implication: The boundary failures are consistent with known fragility of NLU systems under synthetic noise and adversarial or degenerate inputs, and they motivate explicit handling of such cases in preprocessing and routing logic \cite{belinkov2018synthetic,sun2020adversarial}.

## Conclusion
A college-domain assistant is implemented as a controlled conversational pipeline combining ensemble intent inference, emotion detection, scope enforcement, and an LLM response layer with knowledge grounding and operational safeguards. The consolidated literature indicates that transformer-based representations and alignment methods improve capability, but robust deployment still requires explicit scope control, evaluation discipline, and resilience to input noise and distribution shift \cite{vaswani2017attention,devlin2019bert,ouyang2022rlhf,larson2019clinc150,belinkov2018synthetic}.

## References
The bibliography for this report is defined in `Paper/references.bib`.

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
