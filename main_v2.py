import os
import json
import subprocess
import logging
import re
import ast
from typing import Optional, Tuple, Dict, Any, List

# --- INFRASTRUCTURE & LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

MEMORY_FILE = "memory.json"
MAX_ITERATIONS = 15 # Increased for more complex evolution
SHELL_TIMEOUT = 30
MODEL_NAME = "qwen/qwen3.8-27b"

# --- THE MODULAR TOOLSET ---

def list_dir() -> str:
    """List files in the current directory."""
    try:
        files = os.listdir('.')
        return f"Files: {', '.join(sorted(files))}"
    except Exception as e:
        return f"Error: {e}"

def read_file(filename: str) -> str:
    """Read a file safely."""
    try:
        safe_path = os.path.normpath(filename)
        if safe_path.startswith('..') or os.path.isabs(safe_path):
            return "Error: Path traversal attempt blocked."
        with open(safe_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading {filename}: {e}"

def write_file(filename: str, content: str) -> str:
    """Create or overwrite a file."""
    try:
        safe_path = os.path.normpath(filename)
        if safe_path.startswith('..') or os.path.isabs(safe_path):
            return "Error: Path traversal attempt blocked."
        with open(safe_path, 'w') as f:
            f.write(content)
        return f"Success: wrote to {filename}"
    except Exception as e:
        return f"Error writing {filename}: {e}"

def append_file(filename: str, content: str) -> str:
    """Append content to an existing file. CRITICAL for long code blocks."""
    try:
        safe_path = os.path.normpath(filename)
        if safe_path.startswith('..') or os.path.isabs(safe_path):
            return "Error: Path traversal attempt blocked."
        with open(safe_path, 'a') as f:
            f.write(content)
        return f"Success: appended to {filename}"
    except Exception as e:
        return f"Error appending to {filename}: {e}"

def shell_execute(command: str) -> str:
    """Execute shell command with timeout."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=SHELL_TIMEOUT
        )
        return f"OUT: {result.stdout}\nERR: {result.stderr}"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {SHELL_TIMEOUT}s"
    except Exception as e:
        return f"Error: {e}"

# Registry map for the Omnivore Parser
TOOLS = {
    "list_dir": list_dir,
    "read_file": read_file,
    "write_file": write_file,
    "append_file": append_file,
    "shell_execute": shell_execute
}

# --- THE OMNIVORE PARSER (ROBUST VERSION) ---

def parse_action(output: str) -> Optional[Tuple[str, List[str]]]:
    # 1. Keep XML Parsing as it is (it's already robust)
    if "<function=" in output:
        try:
            name = output.split("<function=")[1].split(">")[0].strip()
            params = re.findall(r'<parameter[^>]*>(.*?)</parameter>', output, re.DOTALL)
            return (name, [p.strip() for p in params])
        except Exception: pass

    # 2. REPLACED ACTION: Parsing with AST
    if "ACTION:" in output:
        try:
            line = [l for l in output.split('\n') if "ACTION:" in l][0]
            call = line.split("ACTION:")[1].strip()
            # Use AST to safely evaluate the function call string
            # We wrap it in 'func()' to make it a valid Python expression
            tree = ast.parse(call) 
            if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Call):
                func_name = tree.body[0].value.func.id
                # Convert AST constants back to python values
                args = [ast.literal_eval(arg) for arg in tree.body[0].value.args]
                return (func_name, args)
        except Exception as e:
            logger.error(f"AST Parsing error: {e}")

    return None

# --- AGENT LOOP ---

SYSTEM_PROMPT = """
You are a Senior Systems Engineer Agent.
You MUST use the provided tools to interact with the system.

Available Tools:
- list_dir(): Lists files.
- read_file(filename): Reads a file.
- write_file(filename, content): Overwrites a file.
- append_file(filename, content): Appends to a file (USE THIS for long code).
- shell_execute(command): Runs linux command.

FORMAT:
THOUGHT: [Reasoning]
ACTION: tool_name("arg1", "arg2")
OBSERVATION: [The result]
FINAL ANSWER: [The conclusion]
"""

def run_agent_loop(user_goal):
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'w') as f: json.dump({"user": "Dr. Faulkner"}, f)
    with open(MEMORY_FILE, 'r') as f: memory = json.load(f)

    messages = [{"role": "system", "content": SYSTEM_PROMPT + f"\nMemory: {memory}"},
                {"role": "user", "content": user_goal}]

    for i in range(MAX_ITERATIONS):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=messages, temperature=0.2, max_tokens=1000
            )
            output = response.choices[0].message.content
            print(f"\n--- ITER {i+1} ---\n{output}")

            if "FINAL ANSWER:" in output: return output

            parsed = parse_action(output)
            if parsed:
                tool_name, args = parsed
                if tool_name in TOOLS:
                    obs = TOOLS[tool_name](*args) if args else TOOLS[tool_name]()
                else:
                    obs = f"Error: Tool {tool_name} not found."
                
                print(f"OBSERVATION: {obs}")
                messages.append({"role": "assistant", "content": output})
                messages.append({"role": "user", "content": f"OBSERVATION: {obs}"})
            else:
                messages.append({"role": "user", "content": "No valid ACTION found. Please provide THOUGHT and ACTION."})
        except Exception as e:
            return f"Runtime Error: {e}"

    return "Max iterations reached."

if __name__ == "__main__":
    import sys
    goal = sys.argv[1] if len(sys.argv) > 1 else "System Audit: check environment stability."
    print(run_agent_loop(goal))

