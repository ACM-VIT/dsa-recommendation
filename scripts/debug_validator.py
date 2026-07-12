# scripts/debug_validator.py

from app.validator.response_validator import validate_llm_output

with open("response.txt", "r", encoding="utf-8") as f:
    raw = f.read()

result = validate_llm_output(
    raw_text=raw,
    submission_id="debug",
    source_code=""
)

print(result)
