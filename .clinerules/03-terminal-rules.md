# TERMINAL EXECUTION RULES
1. MANDATORY WRAPPER: Always run commands through `safe_run.py`:
   - Foreground: `python safe_run.py run -- <command>`
   - Background: `python safe_run.py bg --timeout 1800 --name <job_name> -- <command>`
2. CIRCUIT BREAKER: If output contains `CIRCUIT_BREAKER_TRIPPED` or `OOM_HARD_STOP`, STOP IMMEDIATELY. Do not attempt further retries. Output the error and wait for guidance.
3. CLEANUP ON START: At the start of a New Task, run `python safe_run.py cleanup`.