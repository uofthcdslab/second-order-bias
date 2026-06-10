import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
import ast
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

master_df_annotated = pd.read_csv("taget_maped.csv")

prompt_types = ['acceptable_ours', 'non_acceptable_ours']

models_all = ['qwen_instruct', 'qwen_think', 'olmo_instruct', 'olmo_think', 'gpt51_instruct', 'gpt51_think', 'sonnet46_instruct', 'sonnet46_think', 'gemma3_27b_instruct', 'llama31_8b_instruct', 'llama33_70b_instruct', 'phi4_think']

#--------------- first checking llm generation errors

results_path = os.path.join('my_own', 'results_openrouter')
json_files = [pos_json for pos_json in os.listdir(results_path) if pos_json.endswith('.json') and not pos_json.startswith('stats')]

original_errors = []

for model_name in models_all:
    for prompt_name in prompt_types:
        model_file = [this_file for this_file in json_files if model_name in this_file and prompt_name in this_file][0]
        
        with open(os.path.join(results_path, model_file) , 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        for one_instance in data:
            if one_instance['error'] is not None:
                original_errors.append({
                    'model': model_name,
                    'prompt': prompt_name,
                    'data_name': one_instance['data_name'],
                    'id': one_instance['id'],
                    'response': one_instance['response'],
                    'error': one_instance['error']
                })
                
data_errors = set([error['data_name'] for error in original_errors])
model_errors = set([error['model'] for error in original_errors])
id_errors = [error['id'] for error in original_errors]
prompt_errors = set([error['prompt'] for error in original_errors])
data_errors, model_errors, prompt_errors

from collections import Counter

pairs = [(e["data_name"], e["id"]) for e in original_errors]
counts = Counter(pairs)

duplicates = [pair for pair, count in counts.items() if count > 1]
duplicates_full = [e for e in original_errors if (e["data_name"], e["id"]) in duplicates]

# the goal of above section is just to identify responses with errors which cannot be anyway parsed or processed

#----------------- addressing errors by the parser model - due to some error in decoding json containing response and model reasoning

reasoning_present_errors = []

for model_name in models_all:
    model_file = [this_file for this_file in json_files if model_name in this_file][0]
    
    with open(os.path.join(results_path, model_file) , 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for one_instance in data:
        parsed_str = one_instance['parsed'].lower()
        if 'reasoning' not in parsed_str:
            reasoning_present_errors.append({
                'model': model_name,
                'prompt': prompt_name,
                'data_name': one_instance['data_name'],
                'id': one_instance['id'],
                'response': one_instance['response'],
                'parsed': one_instance['parsed'],
                'error': one_instance['error']
            })
            
            
error_tuple = {(e['prompt'], e["data_name"], e["id"]) for e in original_errors}

rows = []
parsing_formatting_errors = []

temp_master = master_df_annotated.set_index(['source', 'tix'])

for model_name in models_all:
    model_file = [this_file for this_file in json_files if model_name in this_file][0]
    
    with open(os.path.join(results_path, model_file) , 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for ix, one_instance in enumerate(data):            
        if one_instance['prompt'] in ['acceptable_ours', 'non_acceptable_ours']:
            dict_key = 'person'
        else:
            dict_key = one_instance['prompt']

        try:
            parsed = json.loads(one_instance['parsed'])
        except Exception as e:
            parsing_formatting_errors.append({
                'model': model_name,
                'prompt': one_instance['prompt'],
                'data_name': one_instance['data_name'],
                'id': one_instance['id'],
                'response': one_instance['response'],
                'parsed': one_instance['parsed'],
                'error': str(e)
            })
            continue

        all_keys = list(parsed.keys())
        for one_key in all_keys:
            if one_key.strip().lower() == 'reasoning':
                dict_key_reason = one_key

        match = temp_master.loc[(one_instance['data_name'], one_instance['id'])]
        
        row = [model_name, one_instance['prompt'], one_instance['data_name'], one_instance['id'], match['text'], match['normalized'], match['normalized_dem_category'], match['type'], match['normalized_secondary'], match['normalized_dem_secondary_category'], parsed[dict_key], parsed[dict_key_reason]]
                
        # adding note for generation errors
        if (one_instance['prompt'], one_instance['data_name'], one_instance['id']) in error_tuple:
            row.append('generation_error')
        else:
            row.append(None)
            
        rows.append(row)
  
# saving parsing errors
error_file = f"my_own/parsing_formatting_errors_new_models.json"
with open(error_file, "w") as f:
    json.dump(parsing_formatting_errors, f, indent=2) 
    
# manually corrected these errors - only a handful

# and then combine the fixed responses with correctly parsed responses
error_file = f"my_own/parsing_formatting_errors_new_models_fixed.json"
with open(error_file , 'r', encoding='utf-8') as file:
    fixed_formatting = json.load(file)
    
for one_instance in fixed_formatting:
    match = temp_master.loc[(one_instance['data_name'], one_instance['id'])]
    parsed = one_instance['parsed']
    if one_instance['prompt'] in ['acceptable_ours', 'non_acceptable_ours']:
        dict_key = 'person'
    else:
        dict_key = one_instance['prompt']
    
    row = [one_instance['model'], one_instance['prompt'], one_instance['data_name'], one_instance['id'], match['text'], match['normalized'], match['normalized_dem_category'], match['type'], match['normalized_secondary'], match['normalized_dem_secondary_category'], parsed[dict_key], parsed["reasoning"]]
    
    rows.append(row)
    
# save the parsed results to csv
parsed_results = pd.DataFrame(rows, columns=['model_name', 'prompt_name', 'data_name', 'id', 'text', 'target_dem_val', 'target_dem_cat', 'target_type', 'target_dem_val_sec', 'target_dem_cat_sec', 'response', 'reasoning', 'gen_error'])
        
parsed_results.to_csv('parsed_results.csv', index=False, encoding='utf-8')

# some more minor formatting to how "Unknown" values were returned
import ast

target_prompts = ["acceptable_ours", "non_acceptable_ours"]

def clean_response_dict(x):
    # Handle actual null response cells
    if pd.isna(x):
        return "Unknown"

    # If response is stored as a string, convert it to a dict
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            return x

    # Only process dictionaries
    if not isinstance(x, dict):
        return x

    cleaned = {
        k: v
        for k, v in x.items()
        if not (
            pd.isna(v)
            or str(v).strip().lower() == "unknown"
        )
    }

    # If every value was Unknown/null
    if len(cleaned) == 0:
        return "Unknown"

    return cleaned


mask = (
    parsed_results["prompt_name"].isin(target_prompts)
    & (parsed_results["gen_error"] == "generation_error")
)

parsed_results.loc[mask, "response"] = parsed_results.loc[mask, "response"].apply(clean_response_dict)

parsed_results.to_csv('parsed_results.csv', index=False, encoding='utf-8')

# finally some minor errors are manually corrected as below
'''
- when generation error was not blank, there were some cases where resposnes contained unknown - done above
- some response column where "person: unknown" instead of just unknown
'''