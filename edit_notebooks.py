"""
Edit both notebooks to fill empty exercise cells.
Requires: Python 3 with json module (stdlib).
"""
import json
import sys

# =============================================================================
# Helpers
# =============================================================================

def load_nb(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_nb(path, nb):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

def find_cell_index(nb, unique_text):
    """Find cell index by a unique substring in source."""
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell.get('source', []))
        if unique_text in src:
            return i
    raise ValueError(f"Cell with text '{unique_text}' not found")

def replace_cell_source(nb, unique_text, new_source_lines, exec_count=None):
    """Replace cell source; preserve outputs if already present."""
    idx = find_cell_index(nb, unique_text)
    nb['cells'][idx]['source'] = new_source_lines
    nb['cells'][idx]['execution_count'] = exec_count
    return idx

def validate_nb(path):
    """Basic structural validation."""
    nb = load_nb(path)
    assert 'cells' in nb
    placeholder_markers = ['# code', 'Code ở đây', '#---> Code', 'Viết code', 'BÀI TẬP', 'EXERCISE']
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = ''.join(cell.get('source', []))
            for marker in placeholder_markers:
                if marker in src.upper() or marker in src:
                    print(f"  WARNING: cell {i} still contains placeholder marker: {marker}")
    return True

def ast_check(path):
    import ast
    nb = load_nb(path)
    errors = 0
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell.get('source', []))
        # skip empty or comment-only cells
        stripped = src.strip()
        if not stripped or stripped.startswith('#'):
            continue
        # skip IPython magics (% / !)
        skip = False
        for line in src.split('\n'):
            if line.strip().startswith(('%', '!')):
                skip = True
                break
        if skip:
            continue
        try:
            ast.parse(src)
        except SyntaxError as e:
            print(f"  SYNTAX ERROR in cell {i}: {e}")
            errors += 1
    if errors == 0:
        print("  All code cells pass AST parse.")
    return errors == 0


# =============================================================================
# 1. Edit Homework notebook (Australian credit)
# =============================================================================

HW_PATH = r'C:\Users\tadod\Repositories\it3190\decisiontree-randomforest\Homework\Decision_tree_Random_Forest_Homework.ipynb'

hw = load_nb(HW_PATH)

# Cell 3: import packages
import_cell = """import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

random_state = 0
"""
replace_cell_source(hw, '# import packages,', [import_cell], exec_count=None)

# Cell 5: load & explore data
data_cell = """data_path = './data/australian.dat'
credit = np.genfromtxt(data_path)
X, y = credit[:, :-1], credit[:, -1]

print("Dữ liệu Australian Credit")
print("Shape:", X.shape, y.shape)
print("Các giá trị nhãn:", np.unique(y))
print("Phân phối nhãn:", {int(label): int(np.sum(y == label)) for label in np.unique(y)})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)
"""
replace_cell_source(hw, "data_path = './data/australian.dat'", [data_cell], exec_count=None)

def dt_max_depth_cell():
    return """cv_scores_by_depth, test_scores_by_depth = [], []
max_depth_values = np.arange(2, 11)

for curr_max_depth in max_depth_values:
    tree = DecisionTreeClassifier(random_state=random_state, max_depth=curr_max_depth)

    # cross-validation
    val_scores = cross_val_score(estimator=tree, X=X_train, y=y_train, cv=5, scoring='f1')
    cv_scores_by_depth.append(val_scores.mean())

    # test
    tree.fit(X_train, y_train)
    curr_pred = tree.predict(X_test)
    test_scores_by_depth.append(f1_score(y_test, curr_pred))

plt.figure()
plt.plot(max_depth_values, cv_scores_by_depth, label='cv', marker='o')
plt.plot(max_depth_values, test_scores_by_depth, label='test', marker='s')
plt.legend()
plt.xlabel('max depth')
plt.ylabel('f1 score')
plt.title('DT validation curve for max_depth')
plt.grid(True)
plt.show()
"""

def dt_param_tuning_cell():
    return """def grid_search(algorithm, n_jobs, dict_param):
    if algorithm == 'decision-tree':
        model = DecisionTreeClassifier(random_state=random_state)
    if algorithm == 'random-forest':
        model = RandomForestClassifier(random_state=random_state)
    classifier = GridSearchCV(estimator=model, cv=5, param_grid=dict_param,
                              n_jobs=n_jobs, scoring='f1')
    classifier.fit(X_train, y_train)
    print('Best model', end=': ')
    print(classifier.best_estimator_)
    return classifier.best_estimator_

def evaluate(model):
    print("Train Accuracy :", accuracy_score(y_train, model.predict(X_train)))
    print("Train f1 score :", f1_score(y_train, model.predict(X_train)))
    print("-" * 50)
    print("Test Accuracy :", accuracy_score(y_test, model.predict(X_test)))
    print("Test f1 score :", f1_score(y_test, model.predict(X_test)))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, model.predict(X_test)))

dict_param = {
    'max_depth': [2, 3, 5, 7, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}
best_tree = grid_search('decision-tree', n_jobs=-1, dict_param=dict_param)
evaluate(best_tree)
"""

def rf_num_trees_cell():
    return """cv_scores_by_trees, test_scores_by_trees = [], []
num_trees = np.arange(5, 151, 10)

for ntrees in num_trees:
    rf = RandomForestClassifier(n_estimators=ntrees, random_state=random_state, n_jobs=-1)
    val_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1')
    cv_scores_by_trees.append(val_scores.mean())
    rf.fit(X_train, y_train)
    curr_pred = rf.predict(X_test)
    test_scores_by_trees.append(f1_score(y_test, curr_pred))

plt.figure()
plt.plot(num_trees, cv_scores_by_trees, label='cv', marker='o')
plt.plot(num_trees, test_scores_by_trees, label='test', marker='s')
plt.legend()
plt.xlabel('n_estimators')
plt.ylabel('f1 score')
plt.title('RF validation curve for n_estimators')
plt.grid(True)
plt.show()
"""

def rf_param_tuning_cell():
    return """dict_param_rf = {
    'max_depth': [3, 5, 10, 15, 20],
    'min_samples_leaf': [1, 5, 10, 20],
    'criterion': ["gini", "entropy"],
    'n_estimators': [50, 100, 150]
}
best_forest = grid_search('random-forest', n_jobs=-1, dict_param=dict_param_rf)
evaluate(best_forest)
"""

# The 4 "# code" cells are at known indices based on the notebook structure.
# We map each by looking at the markdown cells before it.
homework_code_cells = [
    ("###  Khảo sát với các giá trị khác nhau của max_depth", dt_max_depth_cell),
    ("###  Parameter tuning", dt_param_tuning_cell),
    ("### Khảo sát với các giá trị khác nhau của num_trees", rf_num_trees_cell),
    ("### Parameter tuning", rf_param_tuning_cell),
]

def get_markdown_neighbors(nb):
    """Map each code cell to the preceding markdown header."""
    last_md = None
    mapping = {}
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            last_md = ''.join(cell.get('source', [])).strip()
        elif cell['cell_type'] == 'code' and last_md:
            mapping[i] = last_md
    return mapping

mapping = get_markdown_neighbors(hw)
for md_marker, content_func in homework_code_cells:
    for idx, last_md in mapping.items():
        if md_marker in last_md and ''.join(hw['cells'][idx].get('source', [])).strip() == '# code':
            hw['cells'][idx]['source'] = [content_func()]
            hw['cells'][idx]['execution_count'] = None
            break

save_nb(HW_PATH, hw)
print(f"Saved: {HW_PATH}")

# =============================================================================
# 2. Edit Practice notebook (German credit) – fill best_forest evaluation
# =============================================================================

PR_PATH = r'C:\Users\tadod\Repositories\it3190\decisiontree-randomforest\Practice\Decision_tree_Random_Forest.ipynb'

pr = load_nb(PR_PATH)

# Cell 24: evaluate best_forest + learning curve
best_forest_eval_cell = """evaluate(best_forest)

title = 'Learning curve with best forest'
label_curve = {'train': 'train', 'test':'cv'}
plot_learning_curve(best_forest, title, label_curve, X_train, y_train, cv=5)
"""
replace_cell_source(pr, "Đánh giá best_forest và vẽ Learning Curve", [best_forest_eval_cell], exec_count=None)

save_nb(PR_PATH, pr)
print(f"Saved: {PR_PATH}")

# =============================================================================
# Validate both
# =============================================================================
print("\n=== Validation ===")
for p in [HW_PATH, PR_PATH]:
    print(f"\nValidating {p.split('\\')[-1]}:")
    validate_nb(p)
    ast_check(p)

print("\nDone.")
