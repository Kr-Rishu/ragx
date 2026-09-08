from ..config import ANSWER_INPUT_COST_PER_M_TOKENS, ANSWER_OUTPUT_COST_PER_M_TOKENS, USD_TO_INR_RATE

def estimate_cost_usd(prompt_tokens, completion_tokens):
    input_cost = (prompt_tokens / 1_000_000) * ANSWER_INPUT_COST_PER_M_TOKENS
    output_cost = (completion_tokens / 1_000_000) * ANSWER_OUTPUT_COST_PER_M_TOKENS
    return round(input_cost + output_cost, 6)

def usd_to_inr(amount_usd):
    return round(amount_usd * USD_TO_INR_RATE, 4)

def extract_token_usage(langchain_response):
    usage = getattr(langchain_response, 'usage_metadata', None) or getattr(langchain_response, 'response_metadata', {}).get('token_usage', {})
    if isinstance(usage, dict):
        return usage.get('input_tokens', usage.get('prompt_tokens', 0)), usage.get('output_tokens', usage.get('completion_tokens', 0))
    return 0, 0
