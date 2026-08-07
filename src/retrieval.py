from .prompts import REFORMULATION_SYSTEM, REFORMULATION_USER


def parse_reformulations(raw):
    queries = {}
    for line in raw.strip().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key in ("direct", "opposing", "entity"):
            queries[key] = value.strip()
    return queries


def reformulate(llm, claim):
    raw = llm.ask(REFORMULATION_SYSTEM, REFORMULATION_USER.format(claim=claim), max_tokens=150)
    queries = parse_reformulations(raw)
    return {
        "direct": queries.get("direct", claim),
        "opposing": queries.get("opposing", claim),
        "entity": queries.get("entity", claim),
    }


def dedupe(chunks):
    seen = set()
    unique = []
    for c in chunks:
        if c["text"] not in seen:
            seen.add(c["text"])
            unique.append(c)
    return unique


def retrieve_dual(llm, embedder, store, claim, top_k=4):
    queries = reformulate(llm, claim)

    direct_vec = embedder.encode_one(queries["direct"])
    opposing_vec = embedder.encode_one(queries["opposing"])
    entity_vec = embedder.encode_one(queries["entity"])

    supporting = store.search(direct_vec, top_k=top_k)
    opposing = store.search(opposing_vec, top_k=top_k)
    general = store.search(entity_vec, top_k=top_k)

    supporting = dedupe(supporting + [c for c in general if c["score"] > 0.4])
    opposing = dedupe(opposing)

    supporting_texts = {c["text"] for c in supporting}
    opposing = [c for c in opposing if c["text"] not in supporting_texts]

    return {
        "queries": queries,
        "supporting": supporting[:top_k],
        "opposing": opposing[:top_k],
    }
