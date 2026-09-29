"""Offline stand-ins for an LLM reading a research dossier.

They let the full pipeline run without a model or network. Both read only the
sealed dossier documents (never the ML verifier) and cite exact sentences and
reconciled ``signal_score`` values, so their output passes through the same
schema, evidence, and ML gates a real model's output would.

The originator is narrative-first: it weighs catalysts (news, earnings, smart
money, sentiment, valuation) fully and price action at half weight, and it lets
red flags lower conviction without vetoing, which is how an eager analyst
behaves. The ML verifier is what must say no.
"""
from __future__ import annotations

from dataclasses import dataclass

from alpha.agents.models import AnalystReport, AnalystRequest, SourceDocument
from alpha.fund.ideas import IdeaRequest, OriginatorIdea

_WEIGHTS = {
    "news_impact": 1.0, "sentiment": 0.8, "smart_money": 1.0, "earnings": 1.0,
    "valuation": 0.4, "technicals": 0.5,
}


def _skill(document: SourceDocument) -> str:
    parts = document.document_id.split(".")
    return parts[1] if len(parts) >= 3 else ""


def _value(document: SourceDocument, key: str) -> float | None:
    return next((item.value for item in document.numeric_values if item.key == key), None)


def _first_sentence(text: str) -> str:
    head, separator, _ = text.partition(". ")
    sentence = (head + ("." if separator else "")).strip()
    return sentence if len(sentence) >= 12 else text[:200].strip()


@dataclass(frozen=True, slots=True)
class DossierOriginator:
    model_id: str = "fusionfinance-offline-dossier-originator-v1"
    threshold: float = 0.55

    def originate(self, request: IdeaRequest) -> str:
        net, cited, red_flags = 0.0, [], 0
        for document in request.snapshot.documents:
            skill = _skill(document)
            if skill == "risk_review":
                red_flags = sum(
                    1 for sentence in document.text.split(".")
                    if sentence.strip() and "No red flags" not in sentence
                )
                continue
            score = _value(document, "signal_score")
            confidence = _value(document, "signal_confidence")
            if score is None or confidence is None or skill not in _WEIGHTS:
                continue
            contribution = _WEIGHTS[skill] * score * confidence
            net += contribution
            if abs(contribution) > 0.05:
                cited.append(document.document_id)
        if abs(net) < self.threshold or not cited:
            idea = OriginatorIdea(
                ticker=request.ticker, action="pass", conviction=0.0,
                thesis="The dossier does not show enough converging evidence.",
            )
        else:
            sign = 1.0 if net > 0 else -1.0
            conviction = max(0.5, min(0.9, 0.5 + 0.15 * abs(net) - 0.05 * red_flags))
            idea = OriginatorIdea(
                ticker=request.ticker,
                action="long" if sign > 0 else "short",
                horizon_days=5,
                expected_move_bps=sign * (80.0 + 60.0 * min(abs(net), 3.0)),
                conviction=conviction,
                thesis="Independent research signals converge on a catalyst the market has not absorbed.",
                catalysts=("Follow-through on the cited catalyst.",),
                falsifiers=(
                    "Residual return moves against the idea over the horizon.",
                    "Smart money or estimates reverse direction.",
                ),
                cited_document_ids=tuple(cited),
            )
        return idea.model_dump_json()


@dataclass(frozen=True, slots=True)
class DossierAnalystProvider:
    model_id: str = "fusionfinance-offline-dossier-analyst-v1"

    def analyze(self, request: AnalystRequest) -> str:
        documents = tuple(d for d in request.snapshot.documents if request.role in d.roles)
        if not documents:
            documents = request.snapshot.documents[:1]
        weighted = [
            (d, (_value(d, "signal_score") or 0.0) * (_value(d, "signal_confidence") or 0.0))
            for d in documents
        ]
        net = sum(value for _, value in weighted)
        lead = max(weighted, key=lambda item: abs(item[1]))[0]
        direction = "positive" if net > 0.15 else "negative" if net < -0.15 else "neutral"
        if request.role == "risk":
            direction = "neutral"
        score = _value(lead, "signal_score")
        report = AnalystReport(
            role=request.role,
            direction=direction,
            confidence=min(0.9, 0.5 + abs(net) / 2),
            summary=f"The {request.role} evidence in the dossier reads {direction}.",
            citations=({
                "document_id": lead.document_id,
                "quoted_text": _first_sentence(lead.text),
                "numeric_key": "signal_score" if score is not None else "",
                "asserted_value": score,
            },),
            causal_chain=(f"Dossier {request.role} evidence informs the proposed move.",),
            falsifiers=(f"Later {request.role} evidence reverses the observed direction.",),
            risk_flags=(),
            veto=False,
        )
        return report.model_dump_json()


__all__ = ["DossierAnalystProvider", "DossierOriginator"]
