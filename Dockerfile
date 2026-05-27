# Jupyter environment for the IT3190 notebooks (preprocessing/ and regression/).
#
# Python is pinned to 3.11 because scikit-learn 1.1.3 (required by
# regression_homework.ipynb for datasets.load_boston()) does not support 3.12+.
FROM python:3.11-slim

# build-essential is only a fallback in case python-crfsuite (a pyvi dependency) has to
# compile from source; every other package installs from a prebuilt manylinux wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    # pyvi declares an unpinned scikit-learn dep, so install it without deps to keep it from
    # upgrading the pinned scikit-learn==1.1.3. Its real runtime deps (sklearn-crfsuite, ...)
    # are listed explicitly in requirements.txt instead.
    && pip install --no-cache-dir --no-deps pyvi

EXPOSE 8888

# Launch JupyterLab with auth disabled (local-only use; compose binds it to 127.0.0.1).
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--ServerApp.token=", "--ServerApp.password=", \
     "--ServerApp.root_dir=/workspace"]
