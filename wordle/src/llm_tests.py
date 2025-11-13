import argparse
import json
import os
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import wandb

from classes.GameState import GameState, Status
from classes.LetterCell import Feedback
from constants import (
    DEEPSEEK_MODEL,
    LLM_MODEL,
    LLM_PLATFORM,
    MAX_LLM_CONTINUOUS_CALLS,
    OLLAMA_MODEL,
    OPENROUTER_MODEL,
)

LOG_DIR = Path("benchmarks/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_NUM_RUNS = 10
MAX_INVALID_LLM_RESPONSES = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark different LLM configurations on the Wordle agent."
    )
    parser.add_argument(
        "--llm-platform",
        default=LLM_PLATFORM,
        choices=["openai", "ollama", "openrouter", "gemini", "grok", "deepseek"],
        help="Which LLM backend to use for this run.",
    )
    parser.add_argument(
        "--openai-model",
        default=LLM_MODEL,
        help="Model name passed to the OpenAI Chat Completions API.",
    )
    parser.add_argument(
        "--ollama-model",
        default=OLLAMA_MODEL,
        help="Model name served by Ollama for local testing.",
    )
    parser.add_argument(
        "--openrouter-model",
        default=OPENROUTER_MODEL,
        help="Model name when routing via OpenRouter.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DEFAULT_NUM_RUNS,
        help="How many benchmark games to play.",
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=MAX_LLM_CONTINUOUS_CALLS,
        help="Limit on recursive calls when the LLM output is rejected.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path for the JSON results. Defaults to benchmarks/logs/<platform>_<model>_<runs>.json",
    )
    parser.add_argument(
        "--wandb-project",
        default="llm-wordle",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-name",
        default=None,
        help="Optional custom run name in Weights & Biases.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Skip logging metrics to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="Forces WANDB_MODE for this run (online posts immediately).",
    )
    return parser.parse_args()


def resolve_model_name(platform: str, args) -> str:
    match platform:
        case "openai":
            return args.openai_model
        case "ollama":
            return args.ollama_model
        case "openrouter":
            return args.openrouter_model
        case "deepseek":
            return DEEPSEEK_MODEL
        case "gemini":
            return "gemini-2.0-flash"
        case "grok":
            return os.getenv("GROK_MODEL", "x-ai/grok-4-fast:free")
        case _:
            return args.openai_model


def default_log_file(platform: str, model_name: str, num_runs: int) -> Path:
    safe_model = model_name.replace("/", "_").replace(":", "-")
    return LOG_DIR / f"{platform}_{safe_model}_{num_runs}runs.json"


def run_game(
    game: GameState,
    run_id: int,
    totals: dict,
    log_to_wandb: bool,
    results_dict=None,
):
    print(f"Starting run {run_id + 1}")
    game.reset()
    total_completion = 0
    completion = 0
    game_start_time = time.time()
    game_bad_guesses = 0
    game_llm_valid_guesses = 0
    game_llm_calls = 0
    game_llm_latency = 0.0
    game_prompt_chars = 0
    game_solver_fallbacks = 0
    invalid_response_streak = 0

    while game.status != Status.end:
        llm_metrics = game.enter_word_from_ai() or {}
        llm_generated_guess = llm_metrics.get("made_guess", False)
        llm_calls = llm_metrics.get("llm_calls", 0)
        llm_latency = sum(llm_metrics.get("call_latencies", []))
        prompt_chars = sum(llm_metrics.get("prompt_sizes", []))

        game_llm_calls += llm_calls
        totals["llm_calls"] += llm_calls
        game_llm_latency += llm_latency
        totals["llm_latency"] += llm_latency
        game_prompt_chars += prompt_chars
        totals["prompt_chars"] += prompt_chars

        # Handle invalid responses
        if not game.was_valid_guess:
            invalid_response_streak += 1
            totals["bad_guesses"] += 1
            game_bad_guesses += 1
            totals["invalid_llm_responses"] += 1

            if invalid_response_streak >= MAX_INVALID_LLM_RESPONSES:
                print("LLM stuck producing invalid guesses. Falling back to solver for this turn.")
                game.enter_word_from_solver(check=(not game.show_window))
                game.was_valid_guess = True
                totals["solver_fallbacks"] += 1
                game_solver_fallbacks += 1
                invalid_response_streak = 0
                llm_generated_guess = False
            else:
                continue
        else:
            invalid_response_streak = 0

        # get the feedback for the newest locked guess
        offset = 0 if game.status == Status.end else 1
        feedback = game.words[game.current_word_index - offset].get_feedback()

        # check completion
        completion = 0
        for fdb in feedback:
            match fdb:
                case Feedback.incorrect:
                    completion += 0
                case Feedback.present:
                    completion += 0.5
                case Feedback.correct:
                    completion += 1

        total_completion += completion
        if llm_generated_guess:
            game_llm_valid_guesses += 1
            totals["valid_guesses"] += 1

    game_end_time = time.time()
    game_latency = game_end_time - game_start_time
    totals["latency"] += game_latency

    tries_this_game = game.num_of_tries()
    avg_game_completion = total_completion / tries_this_game if tries_this_game else 0
    totals["success"] += 1 if game.success else 0
    totals["tries"] += tries_this_game
    if game.success:
        totals["winning_guess_attempts"] += tries_this_game

    avg_llm_latency_this_game = (game_llm_latency / game_llm_calls) if game_llm_calls else 0.0
    avg_prompt_chars_this_game = (game_prompt_chars / game_llm_calls) if game_llm_calls else 0.0
    invalid_to_valid_ratio_game = (
        game_bad_guesses / game_llm_valid_guesses if game_llm_valid_guesses else None
    )

    print(f"Average game completion: {avg_game_completion} / 5")
    print(f"Average tries: {totals['tries'] / (run_id + 1):.2f}")
    print(f"Average success: {totals['success'] / (run_id + 1):.2f}")
    print(f"Average latency: {totals['latency'] / (run_id + 1):.2f}s")
    print(f"Total bad guesses: {totals['bad_guesses']}")
    print()

    if log_to_wandb:
        avg_latency_per_call = (
            (totals["llm_latency"] / totals["llm_calls"]) if totals["llm_calls"] else 0.0
        )
        avg_llm_calls_per_guess = (
            (totals["llm_calls"] / totals["valid_guesses"]) if totals["valid_guesses"] else 0.0
        )
        invalid_to_valid_ratio = (
            totals["bad_guesses"] / totals["valid_guesses"] if totals["valid_guesses"] else None
        )
        avg_prompt_chars = (
            (totals["prompt_chars"] / totals["llm_calls"]) if totals["llm_calls"] else 0.0
        )
        avg_guesses_per_winning = (
            totals["winning_guess_attempts"] / totals["success"] if totals["success"] else 0.0
        )
        wandb.log(
            {
                "average_game_completion": avg_game_completion,
                "rolling_avg_tries": totals["tries"] / (run_id + 1),
                "rolling_avg_success": totals["success"] / (run_id + 1),
                "rolling_avg_latency": totals["latency"] / (run_id + 1),
                "total_bad_guesses": totals["bad_guesses"],
                "bad_guesses_this_game": game_bad_guesses,
                "llm_calls_this_game": game_llm_calls,
                "avg_llm_latency_per_call": avg_latency_per_call,
                "avg_llm_calls_per_guess": avg_llm_calls_per_guess,
                "invalid_to_valid_ratio": invalid_to_valid_ratio,
                "avg_prompt_size_chars": avg_prompt_chars,
                "avg_guesses_per_winning_game": avg_guesses_per_winning,
            },
            step=(run_id + 1),
        )
   
    if results_dict is not None:
        results_dict["games"].append({
            "run_id": run_id + 1,
            "average_game_completion": avg_game_completion,
            "tries": game.num_of_tries(),
            "success": game.success,
            "latency": game_latency,
            "bad_guesses": game_bad_guesses,
            "valid_llm_guesses": game_llm_valid_guesses,
            "llm_calls": game_llm_calls,
            "avg_llm_latency_per_call": avg_llm_latency_this_game,
            "avg_prompt_size_chars": avg_prompt_chars_this_game,
            "invalid_to_valid_ratio": invalid_to_valid_ratio_game,
            "solver_fallbacks": game_solver_fallbacks,
        })

    return totals



def test_games(game: GameState, num_runs: int, log_file: Path, metadata: dict, log_to_wandb: bool):
    totals = {
        "tries": 0,
        "success": 0,
        "bad_guesses": 0,
        "latency": 0.0,
        "valid_guesses": 0,
        "llm_calls": 0,
        "llm_latency": 0.0,
        "prompt_chars": 0,
        "winning_guess_attempts": 0,
        "invalid_llm_responses": 0,
        "solver_fallbacks": 0,
    }

    results = {
        "num_runs": num_runs,
        "max_llm_continuous_calls": metadata["max_llm_continuous_calls"],
        "llm_platform": metadata["llm_platform"],
        "llm_model": metadata["llm_model"],
        "games": [],
    }

    for i in range(num_runs):
        run_game(
            game=game,
            run_id=i,
            totals=totals,
            log_to_wandb=log_to_wandb,
            results_dict=results,
        )
        if i < num_runs - 1:
            time.sleep(1)

    # Calculate final averages
    win_rate = totals["success"] / num_runs
    avg_tries = totals["tries"] / num_runs
    avg_latency = totals["latency"] / num_runs
    avg_guesses_per_winning = (
        totals["winning_guess_attempts"] / totals["success"] if totals["success"] else 0.0
    )
    avg_llm_calls_per_guess = (
        totals["llm_calls"] / totals["valid_guesses"] if totals["valid_guesses"] else 0.0
    )
    avg_latency_per_llm_call = (
        totals["llm_latency"] / totals["llm_calls"] if totals["llm_calls"] else 0.0
    )
    avg_prompt_chars = (
        totals["prompt_chars"] / totals["llm_calls"] if totals["llm_calls"] else 0.0
    )
    invalid_vs_valid_ratio = (
        totals["bad_guesses"] / totals["valid_guesses"] if totals["valid_guesses"] else None
    )

    # Save the results
    results["total_bad_guesses"] = totals["bad_guesses"]
    results["win_rate"] = win_rate
    results["avg_tries"] = avg_tries
    results["avg_latency"] = avg_latency
    results["avg_guesses_per_winning_game"] = avg_guesses_per_winning
    results["avg_llm_calls_per_guess"] = avg_llm_calls_per_guess
    results["avg_latency_per_llm_call"] = avg_latency_per_llm_call
    results["avg_prompt_size_chars"] = avg_prompt_chars
    results["invalid_to_valid_ratio"] = invalid_vs_valid_ratio
    results["solver_fallbacks"] = totals["solver_fallbacks"]
    results["invalid_llm_responses"] = totals["invalid_llm_responses"]
   
    with open(log_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"FINAL RESULTS:")
    print(f"{'='*50}")
    print(f"Win Rate: {win_rate:.2%} ({totals['success']}/{num_runs})")
    print(f"Average Tries: {avg_tries:.2f}")
    print(f"Average Latency: {avg_latency:.2f}s")
    print(f"Average Guesses (Winning Games): {avg_guesses_per_winning:.2f}")
    print(f"Average LLM Calls per Guess: {avg_llm_calls_per_guess:.2f}")
    print(f"Latency per LLM Call: {avg_latency_per_llm_call:.2f}s")
    print(f"Average Prompt Size: {avg_prompt_chars:.0f} chars")
    if invalid_vs_valid_ratio is None:
        print("Invalid/Valid Guess Ratio: N/A (no valid guesses)")
    else:
        print(f"Invalid/Valid Guess Ratio: {invalid_vs_valid_ratio:.2f}")
    print(f"Total Bad Guesses: {totals['bad_guesses']}")
    print(f"Solver Fallbacks Used: {totals['solver_fallbacks']}")
    print(f"{'='*50}")
    print(f"\nSaved benchmark results to {log_file}")


if __name__ == "__main__":
    args = parse_args()
    model_in_use = resolve_model_name(args.llm_platform, args)
    log_path = args.log_file or default_log_file(args.llm_platform, model_in_use, args.num_runs)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    wandb_mode = args.wandb_mode
    wandb_disabled = args.disable_wandb or wandb_mode == "disabled"
    wandb_run = None
    if not wandb_disabled:
        os.environ["WANDB_MODE"] = wandb_mode
        run_name = args.wandb_name or f"{args.llm_platform}-{model_in_use}-{args.num_runs}runs"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name
        )

    metadata = {
        "llm_platform": args.llm_platform,
        "llm_model": model_in_use,
        "max_llm_continuous_calls": args.max_llm_calls,
    }

    game = GameState(
        show_window=False,
        logging=False,
        llm_platform=args.llm_platform,
        llm_model=args.openai_model,
        ollama_model=args.ollama_model,
        openrouter_model=args.openrouter_model,
        max_llm_calls=args.max_llm_calls,
    )
    test_games(
        game=game,
        num_runs=args.num_runs,
        log_file=log_path,
        metadata=metadata,
        log_to_wandb=wandb_run is not None,
    )
