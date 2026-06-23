from fastapi import APIRouter, Request

from backend.app.models.schemas import RAGRuleCreate, RAGSearchRequest


router = APIRouter(prefix="/api/rag", tags=["rag"])


@router.get("/rules")
def list_rules(request: Request) -> list[dict]:
    return request.app.state.rag_service.list_rules()


@router.post("/rules")
def add_rule(rule: RAGRuleCreate, request: Request) -> dict:
    rule_id = request.app.state.rag_service.add_rule(rule.title, rule.text, rule.tags)
    return {"rule_id": rule_id}


@router.post("/search")
def search_rules(search: RAGSearchRequest, request: Request) -> dict:
    rules = request.app.state.rag_service.search(search.query, search.limit)
    return {"rules": rules, "text": "\n".join(f"- {rule['text']}" for rule in rules)}

