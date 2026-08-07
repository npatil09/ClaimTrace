# ClaimTrace

A claim verification engine built on retrieval-augmented generation, but structured
around a question most RAG projects ignore: what happens when the corpus disagrees
with itself?

## Why this isn't a "chat with your PDF" project

Standard RAG retrieves the top-k most similar chunks to a query and hands them to
an LLM, implicitly assuming they all point the same direction. That's fine for
document Q&A. It falls apart the moment you're checking a *claim* against a body
of sources that might contain outdated reports, contradicted rumors, or plain
disagreement between outlets — which is the normal state of most real-world
corpora (news archives, research literature, internal docs across time).

ClaimTrace retrieves for both sides on purpose:

1. **Query reformulation** - an LLM call rewrites the input claim into three search
   angles: a direct restatement, a negated/opposing framing, and a bare entity
   query. This is the difference between "find things like this claim" and
   "find things that would prove or disprove this claim."
2. **Dual retrieval** - each reformulation hits the vector store separately, so
   supporting and opposing evidence are pulled independently instead of assuming
   whatever's nearest agrees with the claim.
3. **Grounded verdict synthesis** - the LLM is shown both evidence sets and told
   explicitly not to fill gaps with outside knowledge, and to say "insufficient
   evidence" rather than guess. This is deliberate prompt engineering to fight
   the tendency of LLMs to sound confident regardless of evidence quality.
4. **Blended confidence score** - the final confidence isn't just "however sure
   the model says it is." It's 60% the LLM's stated confidence and 40% a
   retrieval-based heuristic (source agreement count + top similarity score), so
   an LLM that's overconfident on thin evidence gets pulled back down.

## Architecture

```
documents --> chunking (sentence-window, overlap) --> embeddings (MiniLM)
                                                              |
                                                              v
                                                        Qdrant vector store
                                                              |
claim --> LLM reformulation --> 3 queries --> dual retrieval (support / oppose)
                                                              |
                                                              v
                                        LLM verdict synthesis + confidence blend
```

## Stack

- **Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2, runs locally, no API cost)
- **Vector DB**: Qdrant, running in embedded/local mode no Docker or server needed
- **LLM**: Groq (Llama 3.3 70B) for reformulation and verdict synthesis free tier, no cost
- **API**: FastAPI, two endpoints (`/ingest`, `/verify`)

## Project structure

```
ClaimTrace/
├── data/
├── src/
├── .gitignore
├── LICENSE
├── README.md
├── demo.py
├── main.py
└── requirements.txt
```

(Adjust this to match your actual filenames before pushing.)

## Setup

```bash
git clone https://github.com/<your-username>/claimtrace.git
cd claimtrace
pip install -r requirements.txt
```

Create a `.env` file or export the key directly:

```bash
export GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

## Running it

```bash
# CLI demo against the included sample corpus
python demo.py

# or run the API
uvicorn main:app --reload
```

The sample corpus in `data/` is four short articles about a fictional battery
launch, written so that two sources agree on the date/specs and a third
contradicts them — that's what the demo claims are built to surface.

## Example output

```
CLAIM: The NovaCell 4 battery will ship in March 2027
  verdict: disputed (confidence: 58)
  reasoning: article_1.txt and article_2.txt both report a March 2027 date, but
  article_3.txt cites a leaked memo claiming late 2027, and notes the source has
  a mixed accuracy record on past leaks.

CLAIM: Nova Dynamics has never missed a product launch date before
  verdict: contradicted (confidence: 71)
  reasoning: article_2.txt explicitly states Nova Dynamics missed two prior
  launch targets, in 2024 and 2025.
```

## What I'd extend next

- Track verdicts over time as new documents get ingested, so a claim's confidence
  can visibly shift when new evidence arrives or a source gets contradicted later
- Swap the heuristic confidence blend for something calibrated against a labeled
  claim/verdict dataset instead of hand-picked weights
- Add a small evidence graph view (which chunks fed which verdict) for auditability

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
