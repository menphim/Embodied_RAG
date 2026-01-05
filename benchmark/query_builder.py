import networkx as nx
import openai
import json
import random
import os
import ipdb
from embodied_nav.config import Config

gml_file_path = './semantic_graphs/enhanced_semantic_graph_semantic_graph_Building99_20241118_160313.gml'

graph = nx.read_gml(gml_file_path)
contexts = []

for node, data in graph.nodes(data=True):
    if 'level' in data and data['level']:
        contexts.append(data['summary'])

def _normalize_base_url(api_base):
    base_url = (api_base or "").rstrip("/")
    if not base_url:
        return None
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    return base_url

def _openrouter_headers(settings):
    headers = {}
    app_url = settings.get("app_url", "").strip()
    app_name = settings.get("app_name", "").strip()
    if app_url:
        headers["HTTP-Referer"] = app_url
    if app_name:
        headers["X-Title"] = app_name
    return headers

def _build_client():
    provider = Config.LLM.get("provider")
    vllm_settings = Config.LLM.get("vllm_settings", {})
    openrouter_settings = Config.LLM.get("openrouter_settings", {})
    if provider == "vllm" or (provider is None and vllm_settings.get("enabled")):
        base_url = _normalize_base_url(vllm_settings.get("api_base"))
        return openai.OpenAI(
            base_url=base_url,
            api_key=vllm_settings.get("api_key"),
        )
    if provider == "openrouter":
        base_url = _normalize_base_url(
            openrouter_settings.get("api_base", "https://openrouter.ai/api")
        )
        api_key_env = openrouter_settings.get("api_key_env", "OPENROUTER_API_KEY")
        api_key = os.getenv(api_key_env)
        headers = _openrouter_headers(openrouter_settings)
        client_kwargs = {"base_url": base_url, "api_key": api_key}
        if headers:
            client_kwargs["default_headers"] = headers
        return openai.OpenAI(**client_kwargs)
    return openai.OpenAI()

client = _build_client()
model_name = Config.LLM["model"]

system_message = {
    "role": "system",
    "content": """
        You are a specialist in designing meaningful and contextually relevant queries for embodied scenarios.
        Your task is to create high-quality queries based on provided descriptions of subareas within a scenario.
        
        ### Types of Queries:
        1. **Explicit**: Direct and specific questions targeting objects or locations (e.g., "Find me the nearest water fountain").
        2. **Implicit**: Indirect questions that require reasoning or contextual understanding (e.g., "Where can I find something to drink?").
        3. **Global**: Broad questions summarizing or analyzing the overall scenario or environment (e.g., "What is the purpose of this space?").

        ### Task Instructions:
        1. Ensure that the queries are aligned with the provided context and presented concisely.
        2. Generate one explicit query, two implicit queries, and one global query relevant to the given scenario.
        3. Queries must reflect natural language and avoid using technical or internal identifiers (e.g., "Cafeteria_ColaRefrigerator_5").

        ### Output Format:
        Provide the queries in JSON format as follows:
        [
            {
                "query": "Generated query text here",
                "type": "explicit"
            },
            {
                "query": "Generated query text here",
                "type": "implicit"
            },
            {
                "query": "Generated query text here",
                "type": "implicit"
            },
            {
                "query": "Generated query text here",
                "type": "global"
            }
        ]

        ### Example Queries:
        - **Explicit**: 
          - "Find me the stairs."
          - "Locate the emergency exit."
          - "Point me to a vending machine."
        - **Implicit**: 
          - "Where can I grab a quick snack?"
          - "Where would someone looking for a comfortable place to rest go?"
          - "Where is a good spot to study quietly?"
        - **Global**: 
          - "What activities does this building support?"
          - "How is the environment designed for accessibility?"
          - "Describe the overall function of the space."
    """
}

idx = 0
query_generation_num = 125
with open("benchmark/data/query.jsonl", 'a') as outfile:
    while idx < query_generation_num:
        context_fragments = random.sample(contexts, 2)
        context_str = "\n".join([f"Subarea {i+1}:\n{frag}" for i, frag in enumerate(context_fragments)])

        messages = [
            system_message,
            {"role": "user", "content": f'Subarea descriptions of the scene:\n"""\n{context_str}\n"""'}
        ]

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )

            if response and response.choices:
                content = response.choices[0].message.content
                content = content.split('```json', 1)[1].split('```')[0]
                parsed_data = json.loads(content)

                for item in parsed_data:
                    outfile.write(json.dumps(item) + '\n')
            print(f"Generation done: {idx}")
            idx += 1
        except Exception as e:
            print(f"Error during generation {idx}: {e}")
