# Running the evaluation (for the report)

Everything below runs from the project root. `<index>` is `data/index/miguelgrinberg__microblog`
(made by `python -m repo_qa.index build https://github.com/miguelgrinberg/microblog`, WITH embeddings).
Results go to `<index>/eval/` unless you pass `--out`.

## 1. Check the question set (10 seconds)
```
python -m repo_qa.eval validate <index>        # expect: 25 questions checked; 0 problem(s).
```
`eval/microblog_questions.json` has 25 questions: 21 answerable, each with the code that answers it ("gold"),
and 4 about things that do not exist in the repository (payments, GraphQL, WebSockets, two-factor login).
Those were verified absent by searching the repository, so the right reply is "I couldn't find this".

## 2. Retrieval: did the right code reach the model? (about a minute, no chat model needed)
```
python -m repo_qa.eval retrieval <index>
```
Compares keyword, vector, hybrid and hybrid+graph. The "What the graph changed" lines list exactly which
questions improved or got worse. This is your main evidence for the knowledge graph.

## 3. Answers: ask every question (slow: one model call per question and condition)
```
python -m repo_qa.eval answers <index> --limit 3          # a quick trial first
python -m repo_qa.eval answers <index>                    # hybrid vs hybrid+graph, all 25 questions
python -m repo_qa.eval answers <index> --resume           # if interrupted (Ctrl+C is safe): continues, retries failures
```
Expect roughly 20 to 60 seconds per answer on a laptop GPU, so 50 runs can take 15 to 50 minutes. Start it and
do something else. Failed questions are recorded with their error and retried by `--resume`.

## 4. Grade by hand (the automatic checks are only signals)
Open `<index>/eval/answers_for_grading.csv` in a spreadsheet. For each row compare the `answer` with
`expected_code` and fill the last-but-one column: **2** = correct, **1** = partly right, **0** = wrong
(for the unanswerable questions, 2 means it correctly said the code does not contain it; inventing an answer is 0).
Grade the same questions under both conditions. If time is short, grade a fixed subset (for example q01 to q13
plus all four unanswerable ones) and say so in the report.

## 5. Build the report tables
```
python -m repo_qa.eval report <index>/eval        # writes report.md (paste the tables into your report)
```

## 6. How accurate are the graph's call edges? (about 15 minutes)
`eval/microblog_call_sample.csv` is a ready-made random sample of 40 call edges between the repository's own functions
(20 certain, 4 likely, 16 guess), each with the line of code where the call happens. For every row ask:
"is this call really made to that function?" and write **y** or **n** in the last column. Then:
```
python -m repo_qa.eval calls-precision eval/microblog_call_sample.csv
```
You get precision per confidence level with a 95% range. With 40 rows the range is wide; say so. To use a different
repository or sample size: `python -m repo_qa.graph <repo> --labeling-csv calls.csv`, then
`python -m repo_qa.eval sample-calls <index> calls.csv --n 40 --out my_sample.csv`.

## What to put in the report
- **Setup**: model (local 8B or Gemini), embedding model, repository, 25 questions in 4 types, the conditions compared.
- **Metrics**: say what each column means (the notes at the end of `report.md` are written for this).
- **Results**: the retrieval table, the by-type table, the "what the graph changed" lists, the answer table, your hand grades.
- **Discussion**: where the graph helped (relationship and multi-file questions), what it cost (more context, slower),
  where it did not help or hurt (look at the "worse" list and the answers that failed), how often citations were invalid.
- **Limits, stated plainly**: Python is dynamic so call edges are approximate (quote your measured precision);
  one repository and 25 questions is a small study; a small local model may ignore instructions sometimes.
