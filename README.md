# Can AI grade essays fairly?

A small reproducible study testing whether large language models — the kind behind ChatGPT — can fairly grade student essays against a standardized rubric. Motivated by a sixth-grader who asked ChatGPT to write a perfect essay, then asked the same ChatGPT to grade it. ChatGPT gave its own essay an 84%.

This repository contains everything needed to rerun the experiment: the essays, the rubrics, the code, and the raw results from 1,120 API calls across four OpenAI models. **You don't need to be a programmer to run it** — see [Running it yourself](#running-it-yourself) below.

The full writeup of findings and what they mean is on Substack. This README is the technical companion.

## What's in this repo

```
artifacts/                  the inputs to the experiment — read these first
  prompt.txt                  the writing prompt students would see
  essay_strong.txt            an essay engineered to score 10/10
  essay_borderline.txt        an essay engineered to be middling
  essay_borderline_gamed.txt  borderline essay dressed up to game the grader
  essay_borderline_injected.txt  borderline essay with prompt-injection attacks
  essay_borderline_l2.txt     borderline essay with non-native-English surface patterns
  rubric_simple.txt           one sentence per scoring dimension
  rubric_complex.txt          full rubric with descriptors at every score point (lenient framing)
  rubric_complex_strict.txt   same structure, deliberately stricter descriptors

results/                    the raw data from every API call (one JSON file per model × essay × rubric)
run_experiment.py           runs N trials of the chosen condition(s) against the OpenAI API
analyze.py                  prints a summary table from any run's JSON output
requirements.txt            Python dependencies
.env.example                template for your API key file
```

## What the study tested

Two essays — one strong, one deliberately middling — were graded by four OpenAI models (`gpt-4o`, `gpt-4o-mini`, `gpt-5.2`, and `gpt-5.2-chat-latest`, which is the model behind free-tier ChatGPT) under four different rubric conditions: no rubric in the prompt, a one-sentence-per-dimension rubric, a full rubric with score-point descriptors written in lenient language, and the same full rubric rewritten with deliberately stricter language. Each condition was run 20 times to measure consistency.

Three additional probes tested the kinds of failures a real classroom deployment would encounter:

- **Adversarial gaming** — a copy of the borderline essay dressed up with sophisticated vocabulary, rubric-keyword echoes, and a fabricated authoritative citation. Same underlying argument; only the presentation changes. Tests whether models can be fooled by surface gaming.
- **Prompt injection** — the borderline essay with embedded instructions aimed at the grader (a fake "[GRADER NOTE: ...]" header, "Ignore previous instructions" mid-essay, a fake footer assigning maximum scores). Tests whether models can be manipulated by text inside the essay.
- **L1/L2 bias** — the borderline essay rewritten with surface patterns characteristic of non-native English writers (article errors, preposition errors, modal/infinitive constructions). Argument structure preserved exactly. Tests whether the model penalizes the writing across rubric domains the surface errors shouldn't affect.

## Headline findings

A frontier reasoning model (`gpt-5.2`) with a written rubric grades essays **at least as consistently as two trained human raters** grading the same paper. On the lenient complex rubric, agreement within one point across 20 trials was essentially 100%; published estimates put trained human raters at 80–95%. So **AI grading isn't intrinsically inconsistent.** It can be quite consistent.

**But consistency isn't the same as fairness.** Three failure modes are visible across this small experiment:

1. **Surface gaming defeats the grader.** Three of four models give a deterministic 10/10 to a borderline argument that's been dressed up with sophisticated vocabulary and a fake citation, no real argument improvement. Switching to a stricter rubric does *not* defend against this — it widens the gap between honest mediocre work (penalized) and gamed mediocre work (still 10/10) from 1.6 to 3.85 points.
2. **Older models honor blatant prompt injection.** `gpt-4o-mini` gives a deterministic 10/10 to a borderline essay that contains the literal text "Ignore previous instructions." `gpt-5.2` ignores the injection completely. The fix is model generation, not rubric strictness.
3. **`gpt-4o` shows L1/L2 bias** — the same argument with non-native-English surface patterns drops by 2 points and the penalty bleeds across rubric domains it shouldn't. The newer `gpt-5.2` family contains the penalty to the Conventions domain, where the rubric warrants it. This is consistent with [Hsieh et al. (2025)](https://arxiv.org/abs/2504.21330), who tested GPT-4o specifically on 25,000 argumentative essays.

For the full story — what this means for the AI-grading products being pitched to school districts, what state assessments are actually using under the hood, and why those are different questions — see the Substack post.

## Running it yourself

Three audiences here. Pick the one that fits.

### If you've never run a Python script before

You'll need three things: an OpenAI account with a few dollars of credit, a working copy of Python on your computer, and roughly half an hour. The whole experiment costs about $8 to reproduce in full.

1. **Get an OpenAI API key.** Sign in (or sign up) at [platform.openai.com](https://platform.openai.com), go to **API keys** in the left menu, click **Create new secret key**, and copy the result somewhere safe — you'll only see it once. You'll also need to add a payment method under **Billing**; $10 of credit is more than enough.

2. **Install Python.** If you're on a Mac, open **Terminal** (Cmd+Space, type "Terminal") and run `python3 --version`. If you see a version number, you're done. If not, download the installer from [python.org](https://www.python.org/downloads/) and run it. On Windows the equivalent is **Command Prompt** or **PowerShell**.

3. **Download this repository.** Click the green **Code** button at the top of the GitHub page and choose **Download ZIP**. Unzip the result somewhere you can find it (your Desktop is fine).

4. **Open a terminal in the folder.** On Mac: open Terminal, type `cd ` (with a trailing space), drag the unzipped folder onto the Terminal window, and press Enter. On Windows: shift-right-click the folder in File Explorer and choose "Open in Terminal."

5. **Save your API key.** In the same terminal, run:
   ```
   echo "OPENAI_API_KEY=sk-paste-your-key-here" > .env
   ```
   Replace `sk-paste-your-key-here` with the key you copied in step 1.

6. **Install the Python packages this code needs.** One command:
   ```
   pip3 install -r requirements.txt
   ```

7. **Run a single small experiment** to confirm everything works. This makes 3 API calls (~3 cents, takes about 30 seconds):
   ```
   python3 run_experiment.py --trials 1 --model gpt-4o --essay essay_borderline.txt --out results/test.json
   ```
   You should see output ending with a small summary table.

8. **See the results** in plain text:
   ```
   python3 analyze.py
   ```

That's it. From here you can change which essay or model is being graded, edit any of the rubrics in `artifacts/` to match your own, or write a new essay and grade that.

### If you want to reproduce the full study

```bash
git clone https://github.com/KarenSpinner/essay-grading-study.git
cd essay-grading-study
echo "OPENAI_API_KEY=sk-..." > .env
pip3 install -r requirements.txt

# Four-condition rubric sweep across both baseline essays (480 calls, ~$3)
for MODEL in gpt-4o gpt-4o-mini gpt-5.2 gpt-5.2-chat-latest; do
  for ESSAY in essay_strong essay_borderline; do
    SAFE=$(echo $MODEL | tr '.-' '_')
    python3 run_experiment.py --trials 20 --model $MODEL --essay ${ESSAY}.txt \
      --temperature default --out "results/run_${ESSAY}_${SAFE}_n20.json"
  done
done

# Strict rubric on baselines (160 calls, ~$1)
for MODEL in gpt-4o gpt-4o-mini gpt-5.2 gpt-5.2-chat-latest; do
  for ESSAY in essay_strong essay_borderline; do
    SAFE=$(echo $MODEL | tr '.-' '_')
    python3 run_experiment.py --trials 20 --model $MODEL --essay ${ESSAY}.txt \
      --conditions prompt_plus_complex_strict --temperature default \
      --out "results/run_${ESSAY}_${SAFE}_strict_n20.json"
  done
done

# Adversarial probes (gamed, injected, L2) under both rubrics (480 calls, ~$4)
for MODEL in gpt-4o gpt-4o-mini gpt-5.2 gpt-5.2-chat-latest; do
  SAFE=$(echo $MODEL | tr '.-' '_')
  for ESSAY in essay_borderline_gamed essay_borderline_injected essay_borderline_l2; do
    python3 run_experiment.py --trials 20 --model $MODEL --essay ${ESSAY}.txt \
      --conditions prompt_plus_complex --temperature default \
      --out "results/run_${ESSAY}_${SAFE}_n20.json"
    python3 run_experiment.py --trials 20 --model $MODEL --essay ${ESSAY}.txt \
      --conditions prompt_plus_complex_strict --temperature default \
      --out "results/run_${ESSAY}_${SAFE}_strict_n20.json"
  done
done

python3 analyze.py
```

### If you just want to read the data

Each file in `results/` is a JSON document with three sections: `config` (which model/essay/temperature was used), `summary` (per-rubric mean, standard deviation, and range across the 20 trials), and `trials` (the full 20 model responses with parsed scores). The summary block is the easiest place to start.

## Methodology details and caveats

- **N = 20** trials per cell. Enough to see the effects reported here, but a publishable replication should use N ≥ 50 across more borderline essays.
- **Temperature = default (1.0)** across all models. This is the only temperature `gpt-5.2-chat-latest` accepts, and it matches what consumer ChatGPT runs at. An earlier exploratory run at temperature 0.7 showed the same qualitative pattern.
- **The rubrics** are modeled on the *structure* of a typical state-assessment writing rubric (three domains, 4+4+2 scoring, score-point descriptors). They are not the verbatim wording of any specific state's rubric.
- **The L2 essay** is a hand-drafted approximation by a non-linguist. It has surface patterns plausibly characteristic of intermediate-to-advanced non-native English writers, but is not validated against L2 writer corpora and does not target a specific L1 background. The per-domain finding for `gpt-4o` aligns with published literature, but the L2 result here should be read as a probe, not a proof.
- **Only argumentative writing** is tested. State writing assessments typically also include informative and narrative writing.
- **Only OpenAI** models are tested. Claude and Gemini may behave differently.
- **The "right" score is unknown.** Without ground-truth scores from trained human raters, this study can only measure self-agreement and cross-condition sensitivity, not accuracy.

## Sources

**Automated essay scoring in state assessments:**
- Cambium Assessment — [ClearSight Automated Essay Scoring FAQ (2020–2021)](https://clearsight.portal.cambiumast.com/content/contentresources/en/ClearSight_Automated-Essay-Scoring_FAQ.pdf)
- Cambium Assessment — [Comparing the Robustness of Automated Scoring Approaches](https://www.cambiumassessment.com/technology/machine-learning/comparing-automated-scoring)
- NCES / Cambium — [NAEP Automated Scoring Challenge results](https://cambiumassessment.com/knowledge-center/news-articles/2023/04/11/12/45/automated-scoring-performance-on-the-naep)

**LLM grading in classrooms:**
- Anthropic — [How educators use Claude](https://www.anthropic.com/news/anthropic-education-report-how-educators-use-claude) (the ~49% automation-heavy grading conversation finding)

**Bias in automated essay scoring:**
- Bridgeman, Trapani & Attali (2012) — [*Comparison of Human and Machine Scoring of Essays: Differences by Gender, Ethnicity, and Country* (Applied Measurement in Education, 25:1)](https://www.tandfonline.com/doi/full/10.1080/08957347.2012.635502)
- Hsieh et al. (2025) — [*Does the Prompt-based Large Language Model Recognize Students' Demographics and Introduce Bias in Essay Scoring?* (arXiv:2504.21330)](https://arxiv.org/abs/2504.21330)

**Adversarial / robustness literature:**
- Les Perelman — [The BABEL Generator and E-Rater (PDF)](https://lesperelman.com/wp-content/uploads/2021/01/Perelman-BABEL-Generator-e-rater.pdf)
- Ding, Riordan et al. — [*Automatic Essay Scoring Systems Are Both Overstable and Oversensitive* (Dialogue & Discourse)](https://journals.uic.edu/ojs/index.php/dad/article/view/12448)

## License

The code in this repository (`run_experiment.py`, `analyze.py`) is released under the MIT License. The essays and rubrics in `artifacts/` are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — you may copy, modify, and redistribute them with attribution.

## Acknowledgements

Study designed with the help of an indignant sixth-grader who started it all. API calls and analysis run with [Claude Code](https://claude.com/claude-code).
