# TASK: Merge Academic Content into Final Paper

You are given:

* Multiple literature review files from different LLMs
* A base file: report.md (contains all sections)
* references.bib

## GOAL:

Create ONE clean, final academic paper inside `paper.md`

## REQUIREMENTS:

### Structure (STRICT):

1. Literature Survey (use at least 5–10 best papers)
2. Methodology (Design)
3. Implementation
4. Results
5. Conclusion
6. References (minimum 25 real citations from references.bib)

---

### Literature Survey Rules:

* Remove duplicates across all review files
* Keep only technically strong explanations
* Maintain logical flow:

  * Traditional methods → Transformers → Chatbots → Intent classification
* Each paper must include:

  * Problem
  * Method
  * Contribution
  * Limitation

---

### Writing Rules:

* Formal academic tone
* No first-person language
* No repetition
* No hallucinated papers
* Only use references from references.bib

---

### LaTeX + Markdown Sync:

* Ensure paper.md and paper.tex are consistent
* Use proper citation keys from references.bib
* Do not break citation format

---

### Output:

* Update ONLY report.md and report.tex
* Keep formatting clean and structured
