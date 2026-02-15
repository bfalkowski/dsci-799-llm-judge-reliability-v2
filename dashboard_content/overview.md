## What is this dashboard?

This dashboard supports **DSCI 799** capstone research on the **reliability of LLM-as-a-Judge evaluation**.
It lets you run experiments, inspect results, and manage data—all in one place.

---------------------------------
Stages Completed
---------------------------------
Stage 1: get dataset from MT-Bench  
    https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/mt_bench/question.jsonl

    https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl
    
    Added to data/raw (do not include in git)

Stage 2: create subset
    Format. Generated manually

        {
        "item_id": "101",
        "question": ".."
        "response": "..."
        },

Stage 3: configure judge

    Get resposne from OPenAI using key

    run_repeated_judging.py to evaluate subset
        format: 
        {"execution_id": GUID,
        "item_id": int, 
        "repeat_idx": int, 
        "judge_model": str, 
        "score": int, 
        "justification": string, 
        "latency_ms": int, 
        "created_at": datetime
        }


Stage 4: run judge on subset 
    run each judge multiple k times

Stage 5: add to dashboard 
    show results

Next steps:  
Add telemetry

derive metrics

