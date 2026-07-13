import asyncio

from app.llm.client import LLMClient
from app.models.domain import LLMPrompt
from app.models.request_schemas import AnalyzeRequest
from app.parser.normalizer import normalize
from app.prompt_builder.builder import build_prompt
from app.rule_engine.engine import run_rules
from tests.fixtures.sample_payloads import VALID_WRONG_ANSWER_PAYLOAD

payload = VALID_WRONG_ANSWER_PAYLOAD.copy()
request = AnalyzeRequest.model_validate(payload)
submission = normalize(request)
outcome = run_rules(submission)

promptOG = build_prompt(submission, outcome)

async def main():
    prompt = LLMPrompt(
        system=promptOG.system,
        user=promptOG.user
    )

    client = LLMClient()
    response = await client.get_structured_completion(prompt)

    print("=" * 80)
    print("RAW MODEL OUTPUT")
    print("=" * 80)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())