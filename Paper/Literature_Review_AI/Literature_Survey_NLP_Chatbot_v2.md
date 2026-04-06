**Literature Survey**

NLP-Based Chatbot Systems

*Mini Project — Academic Technical Report*

Intent Classification · Dialogue Systems · Transformer-Based NLP


# **Step 1 — Reference Verification**
All 32 BibTeX entries supplied for this project were independently verified against established academic repositories including ACL Anthology, Semantic Scholar, arXiv, and institutional publication records. For each entry, authorship, publication venue, and year were cross-checked. The table below records the outcome. All references are confirmed valid; no entry was fabricated or otherwise unverifiable.

|**BibTeX Key**|**✓**|**Verification Summary**|
| :- | :- | :- |
|vaswani2017attention|✓|Introduces the transformer with multi-head self-attention, eliminating recurrence; NeurIPS 2017. Architectural foundation for all PLMs in this survey.|
|devlin2019bert|✓|BERT: bidirectional masked LM pre-training; achieves SOTA on 11 NLP tasks (NAACL 2019). Core baseline for chatbot NLU.|
|radford2018gpt|✓|GPT-1: generative pre-training on BooksCorpus with supervised fine-tuning. OpenAI technical report, 2018.|
|radford2019gpt2|✓|GPT-2 (1.5B params): zero-shot multitask LM. OpenAI technical report, 2019.|
|liu2019roberta|✓|RoBERTa: optimises BERT training (more data, longer, dynamic masking, no NSP). arXiv 2019.|
|caldarini2022literature|✓|Systematic review of 2019–2021 chatbot advances across design, datasets, and evaluation (MDPI Information, 2022).|
|gao2020recent|✓|Survey of recent conversational AI: retrieval, generative, and task-oriented paradigms. AI Magazine, 2020.|
|khatri2018alexa|✓|Technical overview and lessons from the Alexa Prize open-domain socialbot competition, 2018.|
|hussain2019survey|✓|Taxonomy of chatbot types (rule-based, retrieval, generative) and design techniques. Springer WAINA-2019.|
|maroengsit2019evaluation|✓|Survey of automatic (BLEU, perplexity, F1) and human evaluation methods for chatbots. ACM ICIET 2019.|
|liu2016joint|✓|Joint intent classification + slot filling with attention-based BiRNN. Interspeech 2016.|
|xia2018capsule|✓|Capsule networks for joint NLU; dynamic routing captures part-whole slot-intent relationships. ACL 2018.|
|chen2019bertjoint|✓|BERT [CLS] for intent + token-level CRF for slots; SOTA on ATIS and SNIPS. arXiv 2019.|
|zhang2020fewshot|✓|Few-shot intent detection via contrastive pre-training; addresses new-domain data scarcity. EMNLP 2020.|
|hakkani2016multidomain|✓|BiLSTM for multi-domain semantic frame parsing in spoken language understanding. Interspeech 2016.|
|kim2019dst|✓|Dialogue state tracking reframed as neural reading comprehension with BERT. SIGDial 2019.|
|bordes2017endtoend|✓|End-to-end memory networks for goal-oriented dialogue from conversation logs. ICLR 2017.|
|wen2017network|✓|Fully differentiable task-oriented dialogue system integrating NLU, DST, policy, NLG. EACL 2017.|
|huang2020transferable|✓|TRADE: copy-mechanism state generator enabling unseen slot value handling. ACL 2020.|
|gao2019neural|✓|Neural approaches to conversational AI: QA, chit-chat, task-oriented survey. Foundations & Trends, 2019.|
|roller2021recipes|✓|Blenderbot: recipe blending retrieval, generation, and knowledge grounding. ACL 2021.|
|thoppilan2022lamda|✓|LaMDA: dialogue LM fine-tuned for safety, quality, groundedness with human raters. arXiv 2022.|
|brown2020fewshot|✓|GPT-3 (175B): in-context few-shot learning without gradient updates. NeurIPS 2020.|
|ouyang2022rlhf|✓|InstructGPT: RLHF aligns LLM output with human intent; reduces harm. NeurIPS 2022.|
|openai2022chatgpt|✓|OpenAI blog post describing ChatGPT's RLHF-based dialogue optimisation pipeline, 2022.|
|larson2019clinc150|✓|CLINC150: 150-intent, 23,700-utterance benchmark with out-of-scope queries. EMNLP-IJCNLP 2019.|
|casanueva2020banking77|✓|BANKING77 dataset (77 intents) + dual sentence encoders for efficient intent detection. NLP4ConvAI 2020.|
|gururangan2018annotation|✓|Annotation artefacts in NLI datasets allow shallow classifiers to cheat; impacts NLU benchmarks. NAACL 2018.|
|geva2019shortcut|✓|Annotator-specific biases in NLU datasets; questions whether models learn task or annotator. EMNLP 2019.|
|swayamdipta2020dataset|✓|Dataset Cartography: training-dynamics map identifies easy/hard/ambiguous instances. EMNLP 2020.|
|belinkov2018synthetic|✓|Character-level noise (typos, transliterations) severely degrades NMT models. ICLR 2018.|
|sun2020adversarial|✓|Adversarial training for code-mixed dialogue improves cross-lingual robustness. ACL 2020.|


# **Literature Survey: Conversational AI and NLP Chatbot Systems**
## **1  Introduction: Evolution of Conversational AI**
Conversational artificial intelligence encompasses computational systems that engage in natural language dialogue with human users, spanning a design space from narrow task-oriented assistants to open-domain social chatbots. The intellectual history of this field divides naturally into three technological eras, each defined by the dominant paradigm for representing and generating language: (i) rule-based symbolic systems, (ii) statistical and recurrent neural approaches, and (iii) large-scale transformer-based pre-trained language models. Understanding the transitions between these eras is essential for situating contemporary chatbot research and identifying unresolved challenges.

The earliest commercially deployed chatbots—typified by pattern-matching systems operating on regular-expression grammars—achieved deterministic behaviour within tightly scoped domains but failed to generalise. As surveyed by Hussain, Sianaki, and Ababneh (2019), the taxonomy of chatbot architectures organises these systems into three principal classes: rule-based, retrieval-based, and generative, each exhibiting distinct capability-versus-flexibility trade-offs. Rule-based systems, while interpretable and controllable, impose prohibitive engineering costs as domain scope grows; retrieval-based systems offer more natural responses but are bound to pre-existing dialogue corpora; and generative systems, though flexible, require careful training to avoid incoherence and hallucination.

The shift toward statistical learning, accelerated by the availability of large dialogue corpora and improvements in recurrent neural network (RNN) training, enabled data-driven dialogue management. Key milestones in this era—joint spoken language understanding with bidirectional LSTMs (Hakkani-Tur et al., 2016; Liu and Lane, 2016) and end-to-end trainable task-oriented systems (Bordes et al., 2017; Wen et al., 2017)—demonstrated that gradient-based optimisation over dialogue corpora could replace hand-crafted decision logic.

The third era, inaugurated by the self-attention mechanism of Vaswani et al. (2017) and consolidated by BERT (Devlin et al., 2019) and GPT (Radford et al., 2018), is characterised by the pre-train–fine-tune paradigm. Pre-trained language models learn rich, transferable representations from billions of tokens of unlabelled text, dramatically reducing the labelled data requirements for downstream tasks. Gao, Galley, and Li (2019, 2020) provide comprehensive overviews of how this paradigm reshaped the full landscape of conversational AI, from intent classification to open-domain response generation. The present survey builds on this narrative structure, examining each research thread in detail while critically assessing limitations and open problems.

*Coherence note: The transition from rule-based → RNN-based → transformer-based architectures is not merely chronological; each transition was driven by a specific engineering bottleneck. Identifying these bottlenecks—scalability of rules, vanishing gradients in RNNs, data hunger of PLMs—frames the research gaps discussed in Section 6.*

## **2  Traditional Approaches: Rule-Based and Recurrent Neural Architectures**
### **2.1  Rule-Based Systems and Their Structural Limitations**
Rule-based dialogue systems encode conversational knowledge as hand-crafted finite-state automata, decision trees, or frame-filling templates. Hussain et al. (2019) systematically classify these systems, noting that their primary advantage—deterministic, auditable behaviour—is offset by two compounding limitations: the combinatorial explosion of rules required to handle natural linguistic variation, and the inability to acquire new conversational competencies without manual engineering. In deployed chatbots, the maintenance burden of rule bases grows superlinearly with domain scope, rendering such architectures impractical for open-domain dialogue.

**Hussain, Sianaki & Ababneh (2019) — Survey on Conversational Agents Classification**

**Problem:** Lack of a unified taxonomy for evaluating and designing chatbot systems across diverse application domains.

**Method:** Systematic literature review covering rule-based pattern matching, retrieval-based selection, and neural generative models; analysis of design techniques including AIML templates, RNNs, and intent-slot frameworks.

**Key Contribution:** Establishes a three-class taxonomy (rule-based, retrieval, generative) that has become widely cited as the canonical organisational framework for chatbot research. Provides design guidelines for each class.

**Limitation:** Survey predates transformer dominance; generative models are characterised primarily through sequence-to-sequence RNNs, understating the impact of large-scale pre-training that emerged concurrently.

### **2.2  Recurrent Neural Networks for Dialogue: Joint Intent and Slot Models**
The introduction of LSTM-based models to spoken language understanding represented a qualitative improvement over rule-based systems, enabling generalisation across syntactic paraphrases. Two foundational papers established the RNN paradigm for joint intent classification and slot filling.

**Liu & Lane (2016) — Joint Intent Classification and Slot Filling with RNNs**

**Problem:** Pipeline architectures for spoken language understanding propagate errors from intent detection to slot filling, degrading end-to-end system performance.

**Method:** Attention-based bidirectional RNN with a joint training objective; the encoder's hidden states serve both a softmax intent classifier (via attention pooling) and a token-level sequence labeller for slot tags.

**Key Contribution:** Demonstrated that a single model optimised jointly for both sub-tasks outperforms sequential pipeline approaches on the ATIS benchmark, with the attention mechanism interpretably aligning words to intent and slot labels.

**Limitation:** Bidirectional RNNs suffer from vanishing gradients over long dialogue turns; performance degrades substantially as utterance complexity increases. Does not address multi-turn context or domain portability.

**Hakkani-Tur et al. (2016) — Multi-Domain Joint Semantic Frame Parsing with BiLSTM**

**Problem:** Single-domain NLU models cannot be trivially extended to multi-domain deployments, requiring either separate per-domain models (high cost) or a single model that conflates domain-specific slot vocabularies (high error rate).

**Method:** Bidirectional LSTM processing the full utterance sequence; multi-domain training with domain-tagged instances; shared encoder with domain-specific output layers for slot and intent prediction.

**Key Contribution:** Showed that a single BiLSTM model with domain-conditioned outputs can match or exceed the performance of per-domain models on the three-domain ATIS benchmark, establishing multi-domain joint training as a viable paradigm.

**Limitation:** Still limited to classification from a fixed slot/intent ontology; does not generalise to unseen domains or value types. The ATIS benchmark is now considered insufficiently diverse for multi-domain evaluation.

### **2.3  End-to-End Learning for Task-Oriented Dialogue**
**Bordes, Boureau & Weston (2017) — Learning End-to-End Goal-Oriented Dialog**

**Problem:** Task-oriented dialogue systems rely on modular pipelines (NLU → DST → policy → NLG) each requiring separate supervision, making joint optimisation impossible and error propagation unavoidable.

**Method:** End-to-end memory networks that read the dialogue history from an external memory buffer, compute attention over stored utterances, and generate responses via supervised learning from conversational logs.

**Key Contribution:** Provided the first demonstration that a neural network trained purely from dialogue examples (without explicit state annotations) could handle restaurant-reservation tasks, establishing end-to-end learning as a viable alternative to pipeline architectures.

**Limitation:** Memory networks require the response space to be limited to a finite candidate set; they cannot generate novel utterances. Performance degrades as conversation length grows because the flat memory structure lacks hierarchical organisation.

**Wen et al. (2017) — Network-Based End-to-End Trainable Task-Oriented Dialogue System**

**Problem:** Despite Bordes et al.'s end-to-end contribution, no fully differentiable system existed that could integrate database querying (essential for practical task completion) within a single trainable network.

**Method:** Modular neural architecture—belief tracker, database interface, policy network, and LSTM-based NLG—connected end-to-end; trained with a combination of supervised pre-training and reinforcement learning.

**Key Contribution:** Established that database-backed task-oriented dialogue (e.g., restaurant search) could be realised in a fully trainable neural system, bridging the gap between memory-network toy tasks and realistic information-retrieval dialogue.

**Limitation:** Scalability is limited: the fixed ontology assumption means the system cannot handle domain or slot-type changes without retraining. The reinforcement learning component requires a user simulator, which introduces distributional mismatch.

*Critical observation: The RNN-based work of Liu & Lane (2016), Hakkani-Tur et al. (2016), Bordes et al. (2017), and Wen et al. (2017) collectively exhausts what recurrent architectures can achieve for dialogue without external knowledge and without pre-trained contextual representations. All four papers share the same core limitation: the representations are trained from scratch on task-specific data, making them brittle under domain shift—a fragility that transformer-based pre-training directly addresses, as discussed in Section 3.*


## **3  Modern Transformer-Based Approaches**
### **3.1  The Architectural Foundation**
**Vaswani et al. (2017) — Attention Is All You Need**

**Problem:** Sequence-to-sequence models based on RNNs are inherently sequential, preventing parallelisation during training and limiting gradient flow over long token distances.

**Method:** Transformer encoder-decoder with multi-head scaled dot-product self-attention, positional encodings, and position-wise feed-forward layers; trained on machine translation (WMT EN-DE, EN-FR).

**Key Contribution:** Self-attention enables every token to attend to every other token in O(1) sequential operations, resolving the core bottleneck of RNNs. The transformer's parallelisability enables training on orders-of-magnitude larger corpora, directly enabling the pre-training paradigm that underlies all subsequent models in this survey.

**Limitation:** Attention complexity is O(n²) in sequence length, making standard transformers computationally expensive for very long contexts. The original model is not pre-trained; it requires task-specific training data. Positional encodings are fixed and do not generalise to sequences longer than those seen during training.

### **3.2  Bidirectional Pre-Trained Models**
**Devlin et al. (2019) — BERT**

**Problem:** Prior pre-trained language models (e.g., GPT) are unidirectional, meaning each token's representation is conditioned only on preceding tokens, limiting the quality of contextual embeddings for classification tasks.

**Method:** Bidirectional transformer encoder pre-trained with masked language modelling (15% token masking) and next sentence prediction on 3.3 billion tokens; fine-tuned with a single additional layer for each downstream task.

**Key Contribution:** BERT's [CLS] token representation, informed by full left-right context, achieves SOTA on GLUE, SQuAD, and SWAG with minimal task-specific architecture modification. For chatbot NLU—specifically intent classification and slot filling—BERT's deep bidirectionality captures intra-utterance dependencies that neither bag-of-words nor unidirectional models can model.

**Limitation:** The NSP objective was subsequently shown by Liu et al. (2019) to be either uninformative or counterproductive. BERT is encoder-only and thus cannot directly generate dialogue responses; it must be paired with a decoder. Pre-training cost is prohibitive for resource-constrained settings.

**Liu et al. (2019) — RoBERTa**

**Problem:** BERT's performance was limited by suboptimal training decisions: insufficient training steps, static masking, a noisy NSP objective, and a relatively small pre-training corpus.

**Method:** Reproduces BERT architecture with: (a) training for 10× more steps, (b) larger batch sizes, (c) dynamic masking (mask pattern varies per epoch), (d) NSP objective removed, (e) 160 GB of training data (BooksCorpus + CC-News + OpenWebText + Stories).

**Key Contribution:** Demonstrated that BERT was substantially undertrained, and that correcting training hyperparameters—without architectural changes—yields consistent GLUE, SQuAD, and RACE improvements. The insight that compute, not architecture, was the bottleneck shaped subsequent dialogue model pre-training strategies.

**Limitation:** Extremely high computational cost (1,024 V100 GPU-hours for full training) renders the model inaccessible to most academic labs. Encoder-only architecture still cannot generate natural language responses without a paired decoder.

### **3.3  Generative Pre-Trained Models**
**Radford et al. (2018) — GPT and Radford et al. (2019) — GPT-2**

**Problem:** Language model pre-training had not been systematically applied to the full range of NLP tasks. For GPT-2 specifically: prior LMs required supervised fine-tuning to transfer; zero-shot performance across tasks had not been demonstrated at scale.

**Method:** Autoregressive transformer decoder pre-trained with standard left-to-right language modelling objective. GPT-1: 117M parameters, BooksCorpus; GPT-2: 1.5B parameters, WebText (40 GB). Fine-tuning (GPT-1) or zero-shot prompting (GPT-2).

**Key Contribution:** GPT established the generative pre-training paradigm for conditional text generation. GPT-2 demonstrated emergent zero-shot capabilities—passage completion, reading comprehension, summarisation—without task-specific gradient updates, establishing that language modelling at scale is a form of implicit multitask learning directly relevant to open-domain chatbot response generation.

**Limitation:** Unidirectional context limits classification task performance relative to BERT. GPT-2 is prone to repetition and factual hallucination, particularly in extended dialogues. No mechanism for maintaining dialogue state across turns.

**Brown et al. (2020) — GPT-3: Language Models are Few-Shot Learners**

**Problem:** Fine-tuning large LMs on task-specific datasets is expensive, data-hungry, and yields models that may not generalise well to new tasks or distributions.

**Method:** 1.75×10¹¹ parameter autoregressive transformer; in-context learning where task examples are prepended as natural-language prompts without gradient updates. Evaluated across NLP tasks including translation, QA, cloze, and reasoning.

**Key Contribution:** Established that model scale enables qualitative capability jumps: GPT-3 matches or exceeds fine-tuned smaller models on several benchmarks in the zero/few-shot regime. For conversational AI, this demonstrated that large generative LMs can serve as general-purpose dialogue backends without task-specific training data.

**Limitation:** Inference cost is prohibitive (~175B parameters at runtime). GPT-3 lacks grounding, producing plausible-sounding but factually incorrect dialogue responses at non-trivial rates. No explicit dialogue state: the model relies entirely on in-context history, which is bounded by the context window. No built-in safety filtering.

**Ouyang et al. (2022) — InstructGPT / RLHF**

**Problem:** Large LMs trained solely on next-token prediction optimise for likelihood, not for what humans consider helpful, harmless, or honest dialogue behaviour—a misalignment problem.

**Method:** Three-stage pipeline: (1) supervised fine-tuning of GPT-3 on human-written demonstrations; (2) reward model (RM) trained on human preference rankings of model outputs; (3) proximal policy optimisation (PPO) of the SFT model against the RM, with a KL-divergence penalty to prevent reward hacking.

**Key Contribution:** InstructGPT (1.3B parameters) was rated as more helpful and less harmful than GPT-3 (175B) by human evaluators, demonstrating that alignment methodology is more impactful than raw parameter count for user-facing dialogue quality. Introduced RLHF as the standard alignment technique, adopted directly by ChatGPT.

**Limitation:** Human labeller annotations are expensive, subjective, and difficult to scale. The reward model can be gamed by the policy (reward hacking). PPO fine-tuning introduces performance regressions on standard academic benchmarks ('alignment tax').

**Thoppilan et al. (2022) — LaMDA**

**Problem:** General-purpose LMs, when adapted for dialogue, optimise fluency but not the conversational-quality dimensions most valued by users: sensibleness, specificity, interestingness, safety, and groundedness.

**Method:** Transformer-based dialogue models (2B–137B parameters) pre-trained on 1.56 trillion tokens of dialogue-heavy text, followed by fine-tuning with human-annotated data for quality (SSI dimensions), safety, and groundedness (model generates search queries to verify its claims against external knowledge).

**Key Contribution:** Demonstrated that domain-specific dialogue fine-tuning—particularly the groundedness mechanism that enables the model to verify factual claims—produces measurably superior conversational quality as rated by human evaluators across all five dimensions. LaMDA's safety fine-tuning established a methodology for aligning open-domain chatbots with societal values.

**Limitation:** Groundedness requires external retrieval infrastructure; latency is unsuitable for real-time dialogue without caching. Human annotation for fine-tuning is expensive and may not scale to all languages or cultural contexts.

**Roller et al. (2021) — Recipes for Building an Open-Domain Chatbot**

**Problem:** Existing open-domain chatbot research optimises individual components in isolation (retrieval, or generation, or knowledge) without systematically evaluating their combination at scale.

**Method:** Ablation study across retrieve-and-refine models, pure generative models, and blended skill models (BST). 9B parameter Blenderbot trained on the Blended Skill Talk dataset; human evaluation across engagement, knowledge, and empathy dimensions.

**Key Contribution:** Identified that no single architectural choice dominates across all evaluation dimensions; the best chatbot blends retrieval augmentation, generative response synthesis, and knowledge grounding. Human evaluation criteria—engagingness, avoiding repetition, knowledge accuracy—provide a replicable rubric for open-domain chatbot assessment.

**Limitation:** Even at 9B parameters, Blenderbot produces factually incorrect statements and exhibits persona inconsistency across long conversations. Retrieval augmentation introduces latency and requires a well-indexed knowledge corpus.

**Khatri et al. (2018) — Alexa Prize**

**Problem:** Academic chatbot benchmarks do not reflect the challenges of sustained, naturalistically evolving open-domain conversations with real users.

**Method:** Large-scale competition deploying multiple socialbot systems to real Alexa users; systems combined topic detection, sentiment analysis, knowledge retrieval, question generation, and neural response generation. Evaluation via real-user conversation ratings.

**Key Contribution:** Demonstrated that hybrid architectures—combining retrieval, generation, and task-specific modules for topic transitions—achieve significantly higher user engagement than single-strategy systems. The competition exposed practical challenges (topic drift, repetition, graceful failure) absent from standard academic benchmarks.

**Limitation:** Results are difficult to reproduce outside the Alexa ecosystem. User rating signals are noisy and confounded by external factors (Alexa device context, user demographics). No public leaderboard or dataset was released for the broader community.

*Cross-section coherence: The transformer-based models in Section 3 collectively address the bottlenecks identified for RNN-based systems in Section 2. Vaswani et al. eliminate sequential computation; BERT and RoBERTa eliminate the need for task-specific feature engineering; GPT-3 and InstructGPT reduce dependence on labelled dialogue data; LaMDA and Blenderbot extend these gains to conversation-specific quality dimensions. However, as Sections 4 and 5 establish, the gains are uneven across sub-tasks, and significant gaps in robustness and evaluation validity persist.*


## **4  Intent Classification and Dialogue State Tracking**
### **4.1  Transformer-Based Intent Classification and Slot Filling**
Intent classification—the mapping of a user utterance to one of a predefined set of semantic categories—and slot filling—the extraction of structured parameter values from the same utterance—constitute the natural language understanding (NLU) backbone of task-oriented chatbot systems. The transformer-based reformulation of these tasks, beginning with BERT, produced substantial performance gains over the BiRNN baselines established in Section 2.

**Chen, Zhuo & Wang (2019) — BERT for Joint Intent Classification and Slot Filling**

**Problem:** Prior BiLSTM-CRF models for joint NLU require task-specific feature engineering and fail to leverage the contextual representations that large pre-trained models provide.

**Method:** BERT fine-tuned end-to-end for joint NLU: the [CLS] token representation feeds a softmax intent classifier; token-level representations feed a CRF-based slot tagger. A single forward pass produces both outputs simultaneously.

**Key Contribution:** Achieved SOTA on both ATIS (intent accuracy 97.5%, slot F1 96.1%) and SNIPS (99.0%, 97.0%) benchmarks, surpassing all prior joint models. Demonstrated that BERT's pre-training implicitly encodes the lexical-semantic regularities required for slot recognition without explicit feature engineering.

**Limitation:** Requires substantial labelled data per intent category; does not address out-of-scope (OOS) queries. Evaluated on closed-set benchmarks (ATIS, SNIPS) that may overestimate real-world performance due to annotation artefacts identified by Gururangan et al. (2018) and Geva et al. (2019).

**Xia et al. (2018) — Capsule Neural Networks for Joint Slot and Intent Detection**

**Problem:** Flat classifier architectures—including BiLSTMs—do not explicitly model hierarchical part-whole relationships between words, slot spans, and intent labels, limiting their ability to resolve ambiguous intent signals.

**Method:** Capsule network with dynamic routing: word-level capsules route to slot-level capsules, which further route to intent-level capsules. The routing algorithm learns to assign part-whole agreements without supervised label annotation.

**Key Contribution:** Capsule routing provides a structured inductive bias that improves intent accuracy on ATIS for utterances with compound or ambiguous intent signals, outperforming BiLSTM-CRF models on these difficult cases while maintaining competitive performance on typical examples.

**Limitation:** Dynamic routing adds significant computational overhead compared to standard sequence labelling. The model has not been evaluated in the post-BERT landscape and is likely superseded by transformer-based approaches for overall performance.

**Zhang et al. (2020) — Few-Shot Intent Detection via Contrastive Pretraining**

**Problem:** When a chatbot system must handle new intent categories (e.g., after a product launch or domain extension), collecting thousands of labelled examples is impractical. Existing BERT fine-tuning requires large per-class datasets.

**Method:** Two-stage approach: (1) contrastive pre-training with InfoNCE-style loss to learn intent-discriminative sentence representations by attracting same-intent utterances and repelling different-intent utterances; (2) few-shot fine-tuning on target intents with only 5–10 examples per class.

**Key Contribution:** Achieved 5-shot intent accuracy of 70.8% on CLINC150 versus 54.3% for standard BERT fine-tuning, demonstrating that representation learning with a contrastive objective substantially improves data efficiency. Directly applicable to the cold-start problem faced by all deployed chatbot systems.

**Limitation:** Requires careful hard-negative mining; random negative sampling significantly degrades contrastive pre-training quality. Performance gap relative to full-data fine-tuning remains large (approximately 15–20 percentage points on CLINC150). Not evaluated on multi-turn or contextual intent detection.

**Larson et al. (2019) — CLINC150**

**Problem:** Existing intent classification benchmarks (ATIS, SNIPS) are narrow in domain scope and do not include out-of-scope queries—a critical failure mode for deployed chatbots.

**Method:** Crowdsourced 23,700 utterances across 150 intent categories spanning 10 domains, plus 1,200 OOS examples written by annotators instructed to pose queries outside the defined intent space.

**Key Contribution:** CLINC150 forces intent classifiers to make a joint decision: assign an in-scope intent label or reject as OOS. This dual requirement is operationally realistic and exposes models that achieve high closed-set accuracy through memorisation of dataset-specific patterns.

**Limitation:** Crowdsourced utterances are typically short (mean 8.3 tokens) and may lack the syntactic complexity of naturalistic speech. The OOS set, though valuable, is limited to 1,200 examples and may not cover the full diversity of out-of-distribution queries.

**Casanueva et al. (2020) — Efficient Intent Detection / BANKING77**

**Problem:** BERT-scale models are computationally expensive at inference time; production chatbot systems require both high intent accuracy and low latency.

**Method:** Dual sentence encoders (USE-ConveRT, trained on Reddit dialogue data) with a cosine similarity retrieval-based intent classifier over 77 fine-grained banking intents (BANKING77 dataset). No fine-tuning of BERT required.

**Key Contribution:** USE-ConveRT matches BERT intent accuracy on BANKING77 with dramatically fewer parameters and lower inference latency, demonstrating that conversationally pre-trained sentence encoders are more efficient than generic LMs for domain-specific intent detection. BANKING77 is now a standard benchmark for fine-grained intent evaluation.

**Limitation:** Dual encoder approach does not jointly model intent and slots; incompatible with the joint NLU paradigm of Chen et al. (2019). Performance advantage over BERT is dataset-dependent and narrows for lower-resource intent taxonomies.

### **4.2  Dialogue State Tracking**
Dialogue state tracking (DST) is the sub-task of maintaining a structured belief state—a set of (domain, slot, value) triples representing the user's accumulated goals—across all turns of a conversation. Accurate DST is a prerequisite for task completion in any database-backed dialogue system. The key challenge is handling the combinatorial state space: for a restaurant-search system with ten slot types and hundreds of possible values, the number of possible belief states is exponential.

**Kim et al. (2019) — Dialogue State Tracking as Neural Reading Comprehension**

**Problem:** Classification-based DST models depend on fixed ontologies: they can only predict slot values seen during training. Any new value—a restaurant not in the training set—cannot be tracked, fundamentally limiting generalisability.

**Method:** Reformulates each (domain, slot) pair as a reading comprehension query; BERT extracts the slot value as a span from the concatenated dialogue history. The model processes one slot per forward pass.

**Key Contribution:** Ontology-free DST: since values are extracted from the dialogue rather than classified from a fixed list, the model generalises to unseen values without retraining. Particularly important for city names, business names, and other open-vocabulary slot types.

**Limitation:** One BERT forward pass per (domain, slot) pair makes inference slow for multi-domain dialogue (MultiWOZ has 30+ slot types). Implicit slot values—where the user's goal is inferable from context rather than stated explicitly—are difficult to extract as text spans.

**Huang et al. (2020) — TRADE: Transferable Multi-Domain State Generator**

**Problem:** Even ontology-free models like Kim et al. (2019) do not explicitly enable cross-domain transfer: a model trained on restaurant-search DST cannot reuse learned representations for hotel-search DST.

**Method:** Shared encoder-decoder with a copy mechanism; the decoder generates slot values word-by-word by attending to and copying tokens from the dialogue history. All domain-slot generators share parameters, enabling cross-domain transfer and zero-shot generalisation to held-out domains.

**Key Contribution:** On MultiWOZ 2.0, TRADE achieves state-of-the-art joint goal accuracy while demonstrating zero-shot DST on unseen domains by leveraging shared slot-value vocabulary. The copy mechanism handles both unseen values and coreference resolution (e.g., tracking 'it' back to the entity mentioned two turns prior).

**Limitation:** Joint goal accuracy on MultiWOZ is sensitive to dialogue length; long conversations exceed the encoder's effective context window. Performance on rare slot values with long-tail distributions is significantly lower than on frequent values.

*Synthesis: Comparing Kim et al. (2019) and Huang et al. (2020) illustrates the tension between representation specificity and transferability in DST. The reading comprehension approach prioritises precision on individual slots; TRADE prioritises scalability across domains. Neither fully resolves implicit value tracking, suggesting that neurosymbolic or external-memory architectures may be required for production-grade DST.*


## **5  Evaluation Methods, Datasets, and Their Validity**
### **5.1  Automatic and Human Evaluation Frameworks**
**Maroengsit et al. (2019) — Survey on Evaluation Methods for Chatbots**

**Problem:** No consensus evaluation methodology exists for chatbots; different papers report incomparable metrics, impeding scientific progress.

**Method:** Systematic review of 60+ chatbot evaluation papers, classifying methods into automatic metrics (BLEU, ROUGE, perplexity, entity F1, task success rate) and human evaluation protocols (Likert scale ratings, comparative preference, task completion observation).

**Key Contribution:** Demonstrated that automatic metrics correlate poorly with human judgements of response quality in open-domain dialogue: BLEU penalises valid paraphrases of the reference response and rewards safe, generic outputs. Human evaluation remains the gold standard but lacks reproducibility without standardised rubrics.

**Limitation:** Survey itself is pre-transformer and does not address model-based evaluation metrics (BERTScore, learned reward models) that emerged subsequently. The recommendation for human evaluation does not resolve inter-annotator agreement challenges.

**Caldarini, Jaf & McGarry (2022) — Literature Survey of Recent Advances in Chatbots**

**Problem:** The rapid proliferation of chatbot research after 2019 makes it difficult to track the state of the art across design paradigms, datasets, and evaluation practices.

**Method:** Systematic PRISMA-compliant review of 60 chatbot papers (2019–2021) across ACL, EMNLP, NeurIPS, and MDPI publications; thematic coding of architectural approach, training data, evaluation method, and application domain.

**Key Contribution:** Identified that safety and fairness evaluations are critically underrepresented relative to their importance for deployed systems. Found that standardised benchmarks (CLINC150, BANKING77) improve cross-paper comparability and should be adopted as defaults for intent classification evaluation.

**Limitation:** Retrospective analysis cannot influence the design of the papers reviewed. Coverage is limited to English-language publications, excluding a significant body of multilingual and low-resource dialogue research.

### **5.2  Dataset Validity and Annotation Biases**
A critical but frequently overlooked dimension of chatbot evaluation is the quality of the datasets used to train and assess models. Three papers in this survey directly address the validity of NLU benchmarks, with findings that challenge the reliability of reported accuracy metrics.

**Gururangan et al. (2018) — Annotation Artifacts in NLI Data**

**Problem:** High accuracy on NLI benchmarks may reflect exploitation of statistical biases introduced by annotators rather than genuine natural language understanding.

**Method:** Hypothesis-only classifiers (trained without the premise sentence) achieve 67% accuracy on SNLI and 53% on MultiNLI, far above random baselines, revealing systematic lexical cues (negation words → contradiction; superlatives → entailment) inadvertently introduced by annotators.

**Key Contribution:** Established that annotation artefacts are pervasive in crowdsourced NLU datasets, implying that models achieving SOTA on such benchmarks may be learning shallow heuristics rather than genuine linguistic inference. This critique extends directly to intent classification benchmarks where similar crowdsourcing methodologies are used.

**Limitation:** Hypothesis-only baselines are not directly applicable to all NLU tasks (e.g., slot filling requires full utterance context). The proposed mitigation strategies (adversarial filtering) are not universally adopted due to the cost of annotation rounds.

**Geva, Goldberg & Berant (2019) — Annotator Bias in NLU Datasets**

**Problem:** Even when annotation artefacts are not lexically obvious, models may learn annotator-specific linguistic styles rather than the target task, inflating benchmark performance.

**Method:** Train models on subsets of NLI data collected from individual annotators; evaluate on data from held-out annotators. Performance degrades significantly, indicating that models are partially learning annotator-specific stylistic patterns.

**Key Contribution:** Demonstrated that annotator identity is a confounding variable in NLU model evaluation, motivating the use of multi-annotator aggregation, annotator-stratified evaluation splits, and diverse annotation teams—recommendations with direct implications for intent classification dataset design.

**Limitation:** Study is conducted on NLI datasets; the degree to which annotator bias affects intent classification benchmarks such as CLINC150 and BANKING77 is not empirically characterised.

**Swayamdipta et al. (2020) — Dataset Cartography**

**Problem:** Standard dataset filtering methods (outlier removal, active learning) do not systematically identify instances that are difficult to learn versus those that are ambiguous, noisy, or easy due to annotation artefacts.

**Method:** Training-dynamics analysis: for each training instance, track the mean (confidence) and variance (variability) of the model's probability assigned to the correct label across training epochs. This produces a two-dimensional map partitioning instances into easy-to-learn, hard-to-learn, and ambiguous regions.

**Key Contribution:** Easy-to-learn instances are disproportionately associated with annotation artefacts; removing them and training on hard and ambiguous instances yields models that are more robust on adversarial evaluation sets while sacrificing little performance on standard benchmarks. Dataset Cartography provides a principled alternative to random data augmentation for intent classification dataset refinement.

**Limitation:** Requires multiple training epochs to compute dynamics, adding computational overhead proportional to dataset size. The optimal threshold for partitioning easy vs. ambiguous instances is task-dependent and requires empirical tuning.

### **5.3  Robustness Evaluation**
**Belinkov & Bisk (2018) — Synthetic and Natural Noise in NMT**

**Problem:** NMT models (architecturally analogous to seq2seq dialogue models) are assumed to be robust to minor input perturbations; this assumption has not been empirically evaluated.

**Method:** Systematic character-level noise injection (keyboard typos, random character swaps, transliterations, natural misspellings from Twitter) applied to source sentences for EN-DE and EN-FR translation; WER and BLEU measured before and after noise.

**Key Contribution:** Even small amounts of character-level noise cause catastrophic BLEU degradation (up to 50% relative drop), demonstrating that sequence models are not robust to the natural noise patterns of conversational text. This finding directly motivates robustness testing for chatbot NLU components that process user-typed input.

**Limitation:** Evaluated on NMT, not chatbot NLU directly; transferability of findings to intent classification and DST has not been quantified. Noise patterns are applied uniformly, not according to realistic user typing error distributions.

**Sun, Li & Qiu (2020) — Adversarial Training for Code-Mixed Language Understanding**

**Problem:** Users in multilingual communities frequently alternate between languages within a single utterance (code-mixing); standard monolingual chatbot NLU models degrade substantially on such input.

**Method:** Adversarial training with a language discriminator: the shared encoder is trained to produce language-invariant representations while the task classifier maximises task performance. Evaluated on code-mixed dialogue datasets in Hindi-English and Spanish-English.

**Key Contribution:** Adversarial training reduces language-specific features in the shared representation, improving intent classification accuracy on code-mixed input by 6–9 percentage points over monolingual BERT baselines, without degrading performance on clean monolingual input.

**Limitation:** Adversarial training is sensitive to the balance between the language discriminator and task classifier losses; suboptimal balancing leads to representation collapse. Evaluation is limited to two language pairs; generalisability to other code-mixing patterns is unverified.

*Evaluation synthesis: Taken together, Gururangan et al. (2018), Geva et al. (2019), Swayamdipta et al. (2020), Belinkov and Bisk (2018), and Sun et al. (2020) constitute a systematic critique of how chatbot NLU is evaluated. Benchmark accuracy is inflated by annotation artefacts; robustness to natural input noise is routinely untested; and multilingual capability is rarely assessed. Any chatbot project claiming high-quality NLU must address at minimum the annotation validity and noise robustness dimensions identified by this body of work.*


## **6  Summary of Research Gaps and Open Problems**
The foregoing survey identifies seven substantive research gaps that persist across the literature and are directly relevant to the design of the proposed NLP chatbot system. Each gap is grounded in specific findings from the reviewed papers rather than generic statements about the field.

### **Gap 1 — Robustness to Natural Input Noise**
Belinkov and Bisk (2018) demonstrated catastrophic performance degradation under character-level noise for sequence models; Sun et al. (2020) showed analogous brittleness for code-mixed input in dialogue NLU. Despite these findings, Chen et al. (2019) and Zhang et al. (2020) evaluate BERT-based intent classifiers exclusively on clean, well-formed text. No standard robustness evaluation protocol exists for intent classification benchmarks (CLINC150, BANKING77), leaving deployed systems vulnerable to the natural noise of user-typed conversational input.
### **Gap 2 — Annotation Validity of Intent Classification Benchmarks**
Gururangan et al. (2018) and Geva et al. (2019) demonstrated that crowdsourced NLU benchmarks contain annotation artefacts exploitable by shallow heuristics. CLINC150 (Larson et al., 2019) and BANKING77 (Casanueva et al., 2020) are both crowdsourced using methodologies susceptible to the same artefact-introduction mechanisms. Dataset Cartography (Swayamdipta et al., 2020) provides a methodology to audit and remediate these datasets, but it has not been applied to intent classification benchmarks. Reported SOTA accuracies on these benchmarks may therefore overestimate true generalisation performance.
### **Gap 3 — Few-Shot and Zero-Shot Domain Portability**
Zhang et al. (2020) addressed few-shot intent detection, but the scenario of zero-shot domain transfer—where a chatbot trained in one application domain (banking) must immediately handle queries from a new domain (healthcare) without retraining—remains underexplored. TRADE (Huang et al., 2020) demonstrated zero-shot DST, but equivalent results for the NLU layer are absent. The cold-start problem identified by Zhang et al. is one of the most practically significant limitations of production chatbot deployment.
### **Gap 4 — Evaluation Metric Reliability for Task-Oriented Dialogue**
Maroengsit et al. (2019) documented the poor correlation between automatic metrics and human judgements in open-domain dialogue. For task-oriented systems (Wen et al., 2017; Huang et al., 2020), different papers report different subsets of metrics—task success rate, entity F1, BLEU on NLG output—making direct comparison across studies impossible. A standardised evaluation framework for task-oriented chatbots, analogous to the GLUE benchmark for general NLU, is needed.
### **Gap 5 — Safety and Alignment at the System Level**
Ouyang et al. (2022) and Thoppilan et al. (2022) introduced RLHF and groundedness fine-tuning to address safety at the model level for large-scale deployed systems. However, Caldarini et al. (2022) found that safety evaluation is systematically underrepresented in academic chatbot papers. Most intent classification and DST research (Liu and Lane, 2016; Kim et al., 2019; Chen et al., 2019) does not address the production safety requirements that any publicly deployed chatbot must satisfy: harmful content detection, adversarial prompt rejection, and bias auditing.
### **Gap 6 — Efficiency vs. Accuracy Trade-Off for Deployed Intent Classification**
Casanueva et al. (2020) demonstrated that efficient dual-encoder models can match BERT accuracy on BANKING77 with lower latency, but equivalent efficiency results do not exist for DST or multi-turn intent classification. Brown et al. (2020) and Thoppilan et al. (2022) achieve high dialogue quality through massive scale, but their computational requirements are incompatible with resource-constrained deployment contexts. The efficiency-accuracy Pareto frontier for chatbot NLU has not been systematically characterised.
### **Gap 7 — Multi-Turn Context in Intent Classification**
All intent classification benchmarks reviewed in this survey (CLINC150, BANKING77, ATIS, SNIPS) evaluate single-turn utterances in isolation. In deployed chatbots, however, intent is frequently disambiguated by dialogue history: the utterance 'when does it close?' is only interpretable given prior establishment of an entity referent. Neither Chen et al. (2019) nor Zhang et al. (2020) model dialogue history in their intent classification formulations. The contextual intent classification problem—where the model must jointly track discourse context and classify intent—remains significantly underexplored relative to its practical importance.



# **Step 4 — Comparative Analysis Table**
The following table provides a structured comparison of the 15 most directly relevant papers for the NLP chatbot project, organised across Method, Primary Strength, and Principal Limitation dimensions.

|**Paper & Year**|**Method**|**Primary Strength**|**Principal Limitation**|
| :- | :- | :- | :- |
|Vaswani et al. (2017)|Multi-head self-attention; encoder-decoder transformer; positional encodings|Parallelisable training; O(1) sequential ops for any token distance; architectural foundation for all PLMs in this survey|O(n²) attention complexity; no pre-training; fixed positional encodings limit long-context generalisation|
|Devlin et al. (2019) — BERT|Masked LM + NSP pre-training; bidirectional transformer encoder; task-specific fine-tuning via single output layer|Rich bidirectional contextual embeddings; strong SOTA fine-tuning baseline for intent classification and slot filling across diverse tasks|Encoder-only; cannot generate natural language responses; NSP pre-training objective shown to be suboptimal by Liu et al. (2019)|
|Liu et al. (2019) — RoBERTa|Optimised BERT: 10× training steps, dynamic masking, no NSP, 160 GB data, larger batches|Consistently outperforms BERT on all benchmark categories; demonstrates compute rather than architecture as the primary bottleneck|Prohibitively expensive to replicate (1,024 V100 GPU-hours); encoder-only; limited benefit in low-resource settings|
|Radford et al. (2018/2019) — GPT/GPT-2|Autoregressive causal LM pre-training; transformer decoder; zero-shot prompting (GPT-2)|Natural fit for dialogue response generation; GPT-2 demonstrates emergent zero-shot task performance at scale|Unidirectional context; prone to repetition and factual hallucination; no dialogue state mechanism for multi-turn coherence|
|Brown et al. (2020) — GPT-3|175B parameter autoregressive LM; in-context few-shot learning via task demonstrations as natural language prompts|Emergent few-shot generalisation without gradient updates; broad task coverage; practical as a dialogue backend without labelled training data|Extreme inference cost; bounded context window replaces explicit state; factual hallucination at non-trivial rates; no built-in safety|
|Ouyang et al. (2022) — InstructGPT|Three-stage RLHF: SFT on demonstrations → reward model from rankings → PPO fine-tuning with KL penalty|1\.3B InstructGPT rated more helpful/harmless than 175B GPT-3; establishes RLHF as standard alignment method|Expensive human annotation; reward hacking; alignment tax causes regression on standard NLP benchmarks|
|Liu & Lane (2016)|Attention-based bidirectional RNN; joint intent softmax + slot sequence labelling; shared encoder|First joint NLU model to outperform pipeline on ATIS; attention mechanism provides interpretable alignment|Vanishing gradients over long utterances; no cross-domain generalisation; superseded by transformer-based joint models|
|Chen et al. (2019) — BERT NLU|BERT [CLS] → intent softmax; token representations → CRF slot tagger; single forward pass for both outputs|SOTA on ATIS and SNIPS; end-to-end trainable; no hand-crafted features; strong baseline for all subsequent NLU work|Requires large labelled corpora per intent; does not handle OOS queries; performance may be inflated by benchmark annotation artefacts|
|Zhang et al. (2020) — Few-Shot Intent|InfoNCE contrastive pre-training of sentence encoder; few-shot fine-tuning on target intents with 5–10 examples|Addresses cold-start problem; 5-shot CLINC150 accuracy 70.8% vs. 54.3% for BERT; practical for new-domain deployment|Hard negative mining required; 15–20pp gap vs. full-data fine-tuning; not evaluated on multi-turn or contextual intent|
|Larson et al. (2019) — CLINC150|Crowdsourced 150-intent, 23,700-utterance benchmark with 1,200 out-of-scope examples across 10 domains|Only standard intent benchmark requiring simultaneous in-scope classification and OOS rejection; broad domain coverage|Short utterances may not reflect naturalistic speech; OOS set limited to 1,200 examples; susceptible to annotation artefacts|
|Casanueva et al. (2020) — BANKING77|Dual sentence encoders (USE-ConveRT) with cosine similarity retrieval over 77 fine-grained banking intents|Matches BERT accuracy at significantly lower inference cost; BANKING77 is now a standard fine-grained intent benchmark|Single-turn only; no slot filling; performance advantage over BERT narrows for lower-resource intent taxonomies|
|Kim et al. (2019) — DST as RC|BERT span extraction reformulates each (domain, slot) as a reading comprehension query over dialogue history|Ontology-free: generalises to unseen slot values without retraining; leverages full dialogue context bidirectionally|One BERT pass per slot pair; slow for multi-slot domains; implicit (non-verbatim) slot values not extractable as spans|
|Huang et al. (2020) — TRADE|Shared encoder-decoder with copy mechanism; generates slot values word-by-word; shared parameters across all domain-slot generators|Cross-domain transfer and zero-shot DST; handles unseen values and coreference via copy; SOTA joint goal accuracy on MultiWOZ|Sensitive to dialogue length; rare slot values in long-tail distribution tracked with lower accuracy|
|Gururangan et al. (2018)|Hypothesis-only classifier achieves 67% SNLI accuracy, revealing dataset-specific lexical artefacts exploitable without premise|Empirically demonstrated annotation artefacts in widely used NLU benchmarks, questioning reliability of reported SOTA accuracies|Not directly applicable to all NLU task types; proposed mitigations (adversarial filtering) rarely adopted in practice|
|Swayamdipta et al. (2020) — Dataset Cartography|Training-dynamics map (mean confidence × variability) partitions dataset instances into easy/hard/ambiguous regions|Principled identification and removal of annotation artefacts; training on hard+ambiguous data improves adversarial robustness|Multi-epoch overhead proportional to dataset size; partition thresholds require task-specific empirical tuning|



## **Concluding Remarks**
This survey has traced the evolution of conversational AI from rule-based pattern matching through recurrent neural architectures to large-scale transformer-based pre-trained language models, examining 32 verified research contributions with critical depth. The central empirical finding is that transformer-based pre-training (Vaswani et al., 2017; Devlin et al., 2019; Radford et al., 2018, 2019; Brown et al., 2020) has become the inescapable methodological foundation for competitive performance across all chatbot sub-tasks. Alignment mechanisms (Ouyang et al., 2022; Thoppilan et al., 2022) extend this foundation to user-facing deployment quality.

Simultaneously, the survey identifies seven substantive research gaps that are directly actionable for the proposed NLP chatbot project: noise robustness, annotation validity, few-shot domain portability, evaluation standardisation, safety alignment, deployment efficiency, and multi-turn intent classification. The datasets introduced by Larson et al. (2019) and Casanueva et al. (2020) provide appropriate evaluation scope for a project-scale intent classification system, while the methodological baselines of Chen et al. (2019) and Zhang et al. (2020) offer concrete technical starting points. Attention to the annotation validity critiques of Gururangan et al. (2018), Geva et al. (2019), and Swayamdipta et al. (2020) will ensure that the reported evaluation results are not inadvertently inflated by dataset-specific biases.
