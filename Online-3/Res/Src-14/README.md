# Numerical Online 3

For topic explanations and basics, read [TOPICS_README.md](TOPICS_README.md).

## Virtual Environment Setup

A virtual environment keeps the required Python packages for this project separate from the system Python.

### 1. Create a Virtual Environment

Run this from the project folder:

```bash
python3 -m venv .venv
```

On Windows, if `python3` does not work, try:

```bash
python -m venv .venv
```

### 2. Activate the Virtual Environment

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

After activation, your terminal should show something like:

```text
(.venv)
```

### 3. Install Requirements

```bash
python -m pip install -r requirements.txt
```

This installs:

```text
numpy
scipy
matplotlib
```

### 4. Run a Python File

Example:

```bash
python onlines/c.py
```

Another example:

```bash
python random_number/visualize_random_numbers.py
```

### 5. Deactivate the Virtual Environment

When finished:

```bash
deactivate
```

## Quick Full Setup

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```
