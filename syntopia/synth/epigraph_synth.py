import requests
import difflib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from syntopia.parsing.schema import Schema, Variable
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

SEARCH_URL = "https://api.epigraphdb.org/meta/nodes/Gwas/search"


def extract_variable_labels(schema: Schema) -> List[Tuple[str, str]]:
    return [(var.name, var.description) for var in schema.variables]


def gwas_search_fuzzy(text: str, top=5) -> List[Dict[str, str]]:
    search_terms = [text, text.lower()]
    words = text.split()
    if len(words) > 1:
        search_terms.extend([words[0], words[-1]])
    candidates: Dict[str, Dict[str, str]] = {}
    for term in search_terms:
        try:
            r = requests.get(SEARCH_URL, params={"name": term, "limit": top, "full_data": False}, timeout=15)
            r.raise_for_status()
            results = r.json().get("results", [])
            logger.debug(f"Raw API results for '{term}': {results}")
            for hit in results:
                logger.info(f"Hit fields for '{term}': {hit}")
                if "name" in hit and "id" in hit:
                    candidates[hit["name"]] = {"name": hit["name"], "id": hit["id"]}
        except Exception as e:
            logger.warning(f"EpiGraphDB fuzzy search failed for '{term}': {e}")
    logger.info(f"Aggregated candidates for '{text}': {list(candidates.values())}")
    return list(candidates.values())


def gpt_4o_mini_fallback(prompt: str, candidates: List[Dict[str, str]]) -> Dict:
    logger.info(f"GPT-4o-mini fallback called with prompt: {prompt}")
    try:
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set.")
            return {"mapping": None, "correlation": 0.0}
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        answer = response.choices[0].message.content.strip().strip("'\"")
        logger.info(f"GPT-4o-mini raw answer: {answer}")
        mapping = None
        # Try exact match, normalized match, or fuzzy match
        candidate_names = [c["name"] for c in candidates]
        if answer in candidate_names:
            mapping = answer
        else:
            norm_answer = answer.strip().lower()
            norm_candidates = [c.strip().lower() for c in candidate_names]
            if norm_answer in norm_candidates:
                mapping = candidate_names[norm_candidates.index(norm_answer)]
            else:
                close = difflib.get_close_matches(norm_answer, norm_candidates, n=1, cutoff=0.8)
                if close:
                    mapping = candidate_names[norm_candidates.index(close[0])]
        logger.info(f"Final mapping for LLM answer '{answer}': {mapping}")
        return {"mapping": mapping, "correlation": 0.0}
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return {"mapping": None, "correlation": 0.0}


def map_variable_to_ukb(label: str, gpt_fallback, all_labels: List[str]) -> Dict:
    candidates = gwas_search_fuzzy(label.replace("_", " "))
    if candidates:
        prompt = (
            f"Given the variable label: '{label}', and the following candidate GWAS variable names (with IDs): {candidates}, "
            f"and the context of other variables in the dataset: {all_labels}, "
            f"which candidate name is the best match for the label? Respond with the exact candidate name string or 'None'."
        )
        logger.info(f"LLM prompt for '{label}': {prompt}")
        gpt_result = gpt_fallback(prompt, candidates)
        logger.info(f"LLM result for '{label}': {gpt_result}")
        mapping_name = gpt_result.get('mapping')
        mapping_id = None
        if mapping_name:
            for c in candidates:
                if c["name"] == mapping_name:
                    mapping_id = c["id"]
                    break
        logger.info(f"Final mapping for variable '{label}': name={mapping_name}, id={mapping_id}")
        if mapping_name and mapping_id:
            return {"id": mapping_id, "name": mapping_name}
    gpt_result = gpt_fallback(label, [])
    return gpt_result


def fetch_observational_correlation(var1: str, var2: str) -> Optional[float]:
    url = "https://api.epigraphdb.org/api/v1/association/observational/"
    params = {"exposure": var1, "outcome": var2, "resource": "ukb"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            return results[0].get("correlation")
    except Exception as e:
        logger.warning(f"EpiGraphDB correlation fetch failed for {var1}, {var2}: {e}")
    return 0.0


def build_correlation_matrix(schema: Schema, gpt_fallback=gpt_4o_mini_fallback) -> Dict[str, Dict[str, float]]:
    variables = extract_variable_labels(schema)
    all_labels = [label for _, label in variables]
    ukb_mappings = {}
    for name, label in variables:
        mapping = map_variable_to_ukb(label, gpt_fallback, all_labels)
        ukb_mappings[name] = mapping
    matrix = {}
    for i, (name1, _) in enumerate(variables):
        matrix[name1] = {}
        for j, (name2, _) in enumerate(variables):
            if i == j:
                matrix[name1][name2] = 1.0
                continue
            ukb1 = ukb_mappings[name1].get("id")
            ukb2 = ukb_mappings[name2].get("id")
            if ukb1 and ukb2:
                corr = fetch_observational_correlation(ukb1, ukb2)
                matrix[name1][name2] = corr
            else:
                logger.warning(f"Skipping correlation for {name1} and {name2}: missing UKB ID(s) (ukb1={ukb1}, ukb2={ukb2})")
                matrix[name1][name2] = None
    return matrix 