"""Execution engine for Piper agent plans."""

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Callable

import requests

from pipertalk.agent.planner import create_plan, replan
from pipertalk.agent.error_handler import analyze_error, generate_fix, ErrorDecision

LLM_URL = os.environ.get("LLM_URL", "http://192.168.1.16:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3")


def _ollama_chat(system: str, user: str) -> str:
    try:
        resp = requests.post(
            f"{LLM_URL}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")
    except Exception as e:
        print(f"[Executor] Ollama call failed: {e}")
        raise


def _run_generated_code(description: str, speak: Callable | None = None) -> str:
    if speak:
        speak("Writing custom code for this task.")

    home = Path.home()

    system_prompt = (
        "You are an expert Python developer. "
        "Write clean, complete, working Python code. "
        "Use standard library + common packages. "
        "Install missing packages with subprocess + pip if needed. "
        "Return ONLY the Python code. No explanation, no markdown, no backticks.\n\n"
        f"PATHS:\n"
        f"  Home      = r'{home}'\n"
        f"  Desktop   = r'{home / 'Desktop'}'\n"
        f"  Downloads = r'{home / 'Downloads'}'\n"
        f"  Documents = r'{home / 'Documents'}'\n"
    )

    try:
        code = _ollama_chat(system_prompt, f"Write Python code to accomplish this task:\n\n{description}")
        code = re.sub(r"```(?:python)?", "", code).strip().rstrip("`").strip()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        print(f"[Executor] Running generated code: {tmp_path}")

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True,
            timeout=120, cwd=str(Path.home()),
        )

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        output = result.stdout.strip()
        err = result.stderr.strip()

        if result.returncode == 0 and output:
            return output
        elif result.returncode == 0:
            return "Task completed successfully."
        elif err:
            raise RuntimeError(f"Code error: {err[:400]}")
        return "Completed."

    except subprocess.TimeoutExpired:
        raise RuntimeError("Generated code timed out after 120 seconds.")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Generated code failed: {e}")


def _inject_context(params: dict, tool: str, step_results: dict, goal: str = "") -> dict:
    if not step_results:
        return params
    params = dict(params)

    if tool == "file_controller" and params.get("action") in ("write", "create_file"):
        content = params.get("content", "")
        if not content or len(content) < 50:
            all_results = [
                v for v in step_results.values()
                if v and len(v) > 100 and v not in ("Done.", "Completed.")
            ]
            if all_results:
                combined = "\n\n---\n\n".join(all_results)
                params["content"] = combined
                print(f"[Executor] Injected content from previous steps")

    return params


def _call_tool(tool: str, parameters: dict, speak: Callable | None) -> str:
    if tool == "web_search":
        from pipertalk.actions.web_search import web_search
        return web_search(parameters=parameters) or "Done."

    elif tool == "weather_report":
        from pipertalk.actions.weather_report import weather_action
        return weather_action(parameters=parameters) or "Done."

    elif tool == "file_controller":
        from pipertalk.actions.file_controller import file_controller
        return file_controller(parameters=parameters) or "Done."

    elif tool == "send_message":
        from pipertalk.actions.send_message import send_message
        return send_message(parameters=parameters) or "Done."

    elif tool == "reminder":
        from pipertalk.actions.reminder import reminder
        return reminder(parameters=parameters) or "Done."

    elif tool == "youtube_video":
        from pipertalk.actions.youtube_video import youtube_video
        return youtube_video(parameters=parameters) or "Done."

    elif tool == "system_info":
        from pipertalk.actions.system_info import system_info
        return system_info(parameters=parameters) or "Done."

    elif tool == "generated_code":
        desc = parameters.get("description", "")
        if not desc:
            raise ValueError("generated_code requires a 'description' parameter.")
        return _run_generated_code(desc, speak=speak)

    else:
        print(f"[Executor] Unknown tool '{tool}' — falling back to generated_code")
        return _run_generated_code(f"Accomplish this task: {parameters}", speak=speak)


class AgentExecutor:
    MAX_REPLAN_ATTEMPTS = 2

    def execute(
        self,
        goal: str,
        speak: Callable | None = None,
        cancel_flag: threading.Event | None = None,
    ) -> str:
        print(f"\n[Executor] Goal: {goal}")

        replan_attempts = 0
        completed_steps = []
        step_results = {}
        plan = create_plan(goal)

        while True:
            steps = plan.get("steps", [])

            if not steps:
                msg = "I couldn't create a valid plan for this task."
                if speak:
                    speak(msg)
                return msg

            success = True
            failed_step = None
            failed_error = ""

            for step in steps:
                if cancel_flag and cancel_flag.is_set():
                    if speak:
                        speak("Task cancelled.")
                    return "Task cancelled."

                step_num = step.get("step", "?")
                tool = step.get("tool", "generated_code")
                desc = step.get("description", "")
                params = step.get("parameters", {})

                params = _inject_context(params, tool, step_results, goal=goal)

                print(f"\n[Executor] Step {step_num}: [{tool}] {desc}")

                attempt = 1
                step_ok = False

                while attempt <= 3:
                    if cancel_flag and cancel_flag.is_set():
                        break
                    try:
                        result = _call_tool(tool, params, speak)
                        step_results[step_num] = result
                        completed_steps.append(step)
                        print(f"[Executor] Step {step_num} done: {str(result)[:100]}")
                        step_ok = True
                        break

                    except Exception as e:
                        error_msg = str(e)
                        print(f"[Executor] Step {step_num} attempt {attempt} failed: {error_msg}")

                        recovery = analyze_error(step, error_msg, attempt=attempt)
                        decision = recovery["decision"]
                        user_msg = recovery.get("user_message", "")

                        if speak and user_msg:
                            speak(user_msg)

                        if decision == ErrorDecision.RETRY:
                            attempt += 1
                            import time
                            time.sleep(2)
                            continue

                        elif decision == ErrorDecision.SKIP:
                            print(f"[Executor] Skipping step {step_num}")
                            completed_steps.append(step)
                            step_ok = True
                            break

                        elif decision == ErrorDecision.ABORT:
                            msg = f"Task aborted. {recovery.get('reason', '')}"
                            if speak:
                                speak(msg)
                            return msg

                        else:
                            fix_suggestion = recovery.get("fix_suggestion", "")
                            if fix_suggestion and tool != "generated_code":
                                try:
                                    fixed_step = generate_fix(step, error_msg, fix_suggestion)
                                    if speak:
                                        speak("Trying an alternative approach.")
                                    res = _call_tool(
                                        fixed_step["tool"],
                                        fixed_step["parameters"],
                                        speak,
                                    )
                                    step_results[step_num] = res
                                    completed_steps.append(step)
                                    step_ok = True
                                    break
                                except Exception as fix_err:
                                    print(f"[Executor] Fix failed: {fix_err}")

                            failed_step = step
                            failed_error = error_msg
                            success = False
                            break

                if not step_ok and not failed_step:
                    failed_step = step
                    failed_error = "Max retries exceeded"
                    success = False

                if not success:
                    break

            if success:
                return self._summarize(goal, completed_steps, speak)

            if replan_attempts >= self.MAX_REPLAN_ATTEMPTS:
                msg = f"Task failed after {replan_attempts} replan attempts."
                if speak:
                    speak(msg)
                return msg

            if speak:
                speak("Adjusting my approach.")

            replan_attempts += 1
            plan = replan(goal, completed_steps, failed_step, failed_error)

    def _summarize(self, goal: str, completed_steps: list, speak: Callable | None) -> str:
        fallback = f"All done. Completed {len(completed_steps)} steps for: {goal[:60]}."
        try:
            steps_str = "\n".join(f"- {s.get('description', '')}" for s in completed_steps)
            prompt = (
                f'User goal: "{goal}"\n'
                f"Completed steps:\n{steps_str}\n\n"
                "Write a single natural sentence summarizing what was accomplished. "
                "Be direct and positive."
            )
            summary = _ollama_chat(
                "You are a helpful assistant that summarizes completed tasks concisely.",
                prompt,
            )
            if speak:
                speak(summary)
            return summary
        except Exception:
            if speak:
                speak(fallback)
            return fallback
