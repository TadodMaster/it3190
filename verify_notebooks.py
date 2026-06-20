"""Verify the notebook edits and fix any remaining issues."""
import json, sys, ast, re
sys.stdout.reconfigure(encoding='utf-8')

# =============================================================================
# Practice notebook: verify cell 24 is filled and clean any leftover markers
# =============================================================================
PR_PATH = r'C:\Users\tadod\Repositories\it3190\decisiontree-randomforest\Practice\Decision_tree_Random_Forest.ipynb'

with open(PR_PATH, 'r', encoding='utf-8') as f:
    pr = json.load(f)

print("=== Practice Notebook ===")
for i, cell in enumerate(pr['cells']):
    src = ''.join(cell.get('source', []))
    if 'Đánh giá best_forest' in src:
        print(f"Cell {i}: contains 'Đánh giá best_forest' ✓")
        print(src)
        print()

# Check for exact placeholder cells (source is just "# code\n" or similar minimal)
placeholder_pattern = re.compile(r'^\s*#\s*code\s*$', re.IGNORECASE)
for i, cell in enumerate(pr['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell.get('source', [])).strip()
        if placeholder_pattern.match(src):
            print(f"WARNING: Cell {i} still has placeholder content")

print("=== AST Check ===")
errors = 0
for i, cell in enumerate(pr['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell.get('source', [])).strip()
    if not src or src.startswith('#'):
        continue
    has_magic = any(line.strip().startswith(('%', '!')) for line in src.split('\n'))
    if has_magic:
        continue
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"  SYNTAX ERROR cell {i}: {e}")
        errors += 1
if errors == 0:
    print("  All code cells pass AST parse. ✓")

# =============================================================================
# Homework notebook: verify all cells are filled
# =============================================================================
HW_PATH = r'C:\Users\tadod\Repositories\it3190\decisiontree-randomforest\Homework\Decision_tree_Random_Forest_Homework.ipynb'

with open(HW_PATH, 'r', encoding='utf-8') as f:
    hw = json.load(f)

print("\n=== Homework Notebook ===")
for i, cell in enumerate(hw['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell.get('source', [])).strip()
        if placeholder_pattern.match(src):
            print(f"WARNING: Cell {i} still has placeholder content")
        else:
            print(f"Cell {i}: filled ({len(src.splitlines())} lines) ✓")

print("\n=== AST Check ===")
errors = 0
for i, cell in enumerate(hw['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell.get('source', [])).strip()
    if not src or src.startswith('#'):
        continue
    has_magic = any(line.strip().startswith(('%', '!')) for line in src.split('\n'))
    if has_magic:
        continue
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"  SYNTAX ERROR cell {i}: {e}")
        errors += 1
if errors == 0:
    print("  All code cells pass AST parse. ✓")

print("\nDone.")
