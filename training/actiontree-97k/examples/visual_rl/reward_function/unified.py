# -*- coding: utf-8 -*-
"""
Unified Reward Function - task routing.
Automatically call the corresponding reward module based on problem_type.
"""

import os
import sys
import logging
from typing import Any, Dict, List

# Add this directory to sys.path to support dynamic loading.
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from utils import normalize_response, preprocess_ground_truth
import math_task as math_reward
import multiple_choice as mc_reward
import numerical as num_reward
import open_ended as oe_reward
import ocr as ocr_reward
import boolean as bool_reward
import code_task as code_reward
import llava as llava_reward
import grounding as grounding_reward

logger = logging.getLogger(__name__)

REWARD_NAME = "unified"
REWARD_TYPE = "batch"

# Map task types to reward modules.
REWARD_MAPPING = {
    # Math tasks.
    "math": math_reward,
    "mathematics": math_reward,

    # Multiple-choice tasks.
    "multiple choice": mc_reward,
    "multiple_choice": mc_reward,

    # Numerical tasks.
    "numerical": num_reward,
    "number": num_reward,
    "regression": num_reward,

    # Open-ended tasks.
    "open-ended": oe_reward,
    "open_ended": oe_reward,
    "video qa": oe_reward,
    "video description": oe_reward,
    "free-form": oe_reward,

    # OCR
    "ocr": ocr_reward,

    # Boolean tasks.
    "boolean": bool_reward,
    "binary classification": bool_reward,

    # LLaVA Critic
    "llava": llava_reward,
    "critic": llava_reward,

    # Code tasks.
    "code": code_reward,
    "coding": code_reward,
    "svg-code": code_reward,
    "html-code": code_reward,

    # Grounding tasks with special handling.
    "spatial grounding": "spatial",
    "temporal grounding": "temporal",
    "spatial-temporal grounding": "spatial-temporal",
}


def compute_score(reward_inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, float]]:
    """
    Unified entry point for reward computation.
    Route each sample to its reward function based on problem_type.
    """
    results = []

    for idx, inp in enumerate(reward_inputs):
        try:
            # Preprocess the inputs.
            response = normalize_response(inp.get("response", ""))
            ground_truth = preprocess_ground_truth(inp.get("ground_truth", ""))
            problem_type = (inp.get("problem_type", "") or "").lower().strip()

            # Update inp.
            inp_processed = {**inp, "response": response, "ground_truth": ground_truth}

            # Get the corresponding reward module or type.
            reward_target = REWARD_MAPPING.get(problem_type, oe_reward)

            # Handle grounding tasks separately.
            if reward_target == "spatial":
                score_list = grounding_reward.spatial_compute_score([inp_processed], **kwargs)
            elif reward_target == "temporal":
                score_list = grounding_reward.temporal_compute_score([inp_processed], **kwargs)
            elif reward_target == "spatial-temporal":
                score_list = grounding_reward.spatial_temporal_compute_score([inp_processed], **kwargs)
            else:
                # Call compute_score in the corresponding module.
                score_list = reward_target.compute_score([inp_processed], **kwargs)

            results.append(score_list[0])

        except Exception as e:
            logger.error(f"Error computing reward for sample {idx}: {e}")
            results.append({
                "overall": 0.0,
                "accuracy": 0.0,
                "format": 0.0,
                "length_penalty": 0.0,
            })

    return results
