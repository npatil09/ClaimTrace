import re
from .prompts import VERDICT_SYSTEM, VERDICT_USER, format_evidence
from .retrieval import retrieve_dual


def parse_verdict(raw):
    verdict_match = re.search(r'verdict:\s*([^\n]+)', raw, re.IGNORECASE)
    confidence_match = re.search(r'confidence:\s*(\d+)', raw, re.IGNORECASE)
    reasoning_match = re.search(r'reasoning:\s*(.+)', raw, re.IGNORECASE | re.DOTALL)

    return {
        "verdict": verdict_match.group(1).strip().lower() if verdict_match else "unknown",
        "llm_confidence": int(confidence_match.group(1)) if confidence_match else 0,
        "reasoning": reasoning_match.group(1).strip() if reasoning_match else raw.strip(),
    }


def retrieval_confidence(evidence):
    supporting = evidence["supporting"]
    opposing = evidence["opposing"]

    if not supporting and not opposing:
        return 0.0

    sources_support = {c["source"] for c in supporting}
    sources_oppose = {c["source"] for c in opposing}

    agreement = len(sources_support) - len(sources_oppose)
    top_score = max([c["score"] for c in supporting] + [0])

    raw = (agreement * 15) + (top_score * 40)
    return max(0.0, min(100.0, raw + 30))


def verify_claim(llm, embedder, store, claim, top_k=4):
    evidence = retrieve_dual(llm, embedder, store, claim, top_k=top_k)

    prompt = VERDICT_USER.format(
        claim=claim,
        supporting=format_evidence(evidence["supporting"]),
        opposing=format_evidence(evidence["opposing"]),
    )
    raw = llm.ask(VERDICT_SYSTEM, prompt, max_tokens=400)
    parsed = parse_verdict(raw)

    r_conf = retrieval_confidence(evidence)
    blended = round((parsed["llm_confidence"] * 0.6) + (r_conf * 0.4))

    return {
        "claim": claim,
        "verdict": parsed["verdict"],
        "confidence": blended,
        "reasoning": parsed["reasoning"],
        "supporting_evidence": evidence["supporting"],
        "opposing_evidence": evidence["opposing"],
        "queries_used": evidence["queries"],
    }
