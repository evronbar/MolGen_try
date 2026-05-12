import os
import re
import importlib.util

def get_imports():
    imports = set()
    for root, dirs, files in os.walk("."):
        if 'venv' in root or '.conda' in root: continue
        for file in files:
            if file.endswith(".py"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    for line in f:
                        match = re.match(r"^(?:from|import)\s+([\w\d_]+)", line)
                        if match:
                            imports.add(match.group(1))
    return imports

internal = {'molgen', 'scripts', 'tests', 'data', 'utils'}
std_libs = {'os', 'sys', 're', 'time', 'math', 'argparse', 'json', 'random', 'datetime', 'typing', 'pathlib', 'abc', 'copy', 'functools', 'itertools', 'collections', 'threading', 'pickle', 'logging', 'enum', 'runpy', 'importlib'}

print("🔍 Scanning project for dependencies...")
all_imports = get_imports()
external_deps = all_imports - internal - std_libs

missing = []
for dep in sorted(external_deps):
    if importlib.util.find_spec(dep) is None:
        missing.append(dep)

if missing:
    print(f"❌ Missing libraries: {missing}")
else:
    print("✅ All external dependencies are installed!")
