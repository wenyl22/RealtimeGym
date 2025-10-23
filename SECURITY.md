# Security Guide: Managing API Keys Safely

This guide explains how to store and use API keys securely in RealtimeGym to prevent accidental commits to version control.

## Quick Start

### 1. Install python-dotenv (optional but recommended)

```bash
pip install python-dotenv
```

This allows automatic loading of `.env` files. If not installed, you can still use system environment variables.

### 2. Create a `.env` file

Copy the example template:

```bash
cp .env.example .env
```

### 3. Add your API keys to `.env`

Edit `.env` and replace the placeholder values:

```bash
# OpenAI API Key (for GPT models)
OPENAI_API_KEY=sk-proj-your-actual-key-here

# DeepSeek API Key
DEEPSEEK_API_KEY=your-deepseek-key-here

# Google/Gemini API Key
GOOGLE_API_KEY=your-google-key-here
```

### 4. Use environment variables in config files

Your YAML config files should reference environment variables using `${VAR_NAME}` syntax:

```yaml
model: gpt-5-mini
api_key: ${OPENAI_API_KEY}
inference_parameters:
  max_tokens: 8192
  reasoning_effort: low
```

### 5. Run agile_eval as usual

```bash
agile_eval --game freeway \
    --mode reactive \
    --reactive-model-config configs/example-gpt-5-mini-reactive.yaml \
    --settings freeway_E_8192_reactive_4096
```

The API keys will be automatically loaded from your `.env` file.

---

## Multiple Approaches

### Approach 1: Using `.env` File (Recommended)

**Pros:**
- Easy to manage multiple keys
- Automatically loaded by the application
- Clear separation of secrets from code
- Works across all projects

**Setup:**

1. Create `.env` in project root:
```bash
OPENAI_API_KEY=sk-proj-...
DEEPSEEK_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
```

2. Reference in YAML configs:
```yaml
api_key: ${OPENAI_API_KEY}
```

3. The `.env` file is already in `.gitignore`, so it won't be committed.

### Approach 2: System Environment Variables

**Pros:**
- No additional files needed
- Works system-wide
- Most secure for production environments

**Setup:**

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
export OPENAI_API_KEY="sk-proj-..."
export DEEPSEEK_API_KEY="sk-..."
export GOOGLE_API_KEY="AIza..."
```

Then reload:
```bash
source ~/.bashrc  # or ~/.zshrc
```

### Approach 3: Personal Config Files

**Pros:**
- Keep separate configs for different models/experiments
- Can version control example configs safely

**Setup:**

1. Create personal configs with `-personal` or `my-` prefix:
```bash
cp configs/example-gpt-5-mini-reactive.yaml configs/gpt-5-mini-personal.yaml
```

2. Add your actual API key:
```yaml
model: gpt-5-mini
api_key: sk-proj-your-actual-key
inference_parameters:
  max_tokens: 8192
```

3. These files are automatically ignored by git (see `.gitignore`):
```
configs/*-personal.yaml
configs/my-*.yaml
```

---

## What's Protected by .gitignore

The following patterns are automatically ignored and safe for storing API keys:

```
# Environment files
.env
.env.local
.env.*.local

# Personal config files
configs/*-personal.yaml
configs/my-*.yaml
configs/deepseek*
```

---

## Best Practices

### DO:
- ✅ Use `.env` files for local development
- ✅ Use environment variables for production/servers
- ✅ Create personal config files with `-personal` or `my-` prefix
- ✅ Always check files before committing: `git diff`
- ✅ Use `${VAR_NAME}` syntax in YAML configs
- ✅ Keep `.env.example` updated with variable names (but not values!)

### DON'T:
- ❌ Commit `.env` files to git
- ❌ Put API keys directly in example config files
- ❌ Share `.env` files in chat/email
- ❌ Use production keys in development
- ❌ Commit files with `api_key: sk-...` or similar

---

## Verifying Your Setup

### Check if .env is loaded

```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OPENAI_API_KEY:', os.getenv('OPENAI_API_KEY')[:10] + '...' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"
```

### Check what files would be committed

```bash
git status
git diff configs/
```

Make sure no files with real API keys appear!

### Test your config

```bash
# This will fail fast if environment variables are missing
agile_eval --game freeway \
    --mode reactive \
    --reactive-model-config configs/example-gpt-5-mini-reactive.yaml \
    --settings freeway_E_8192_reactive_4096 \
    --seed_num 1 \
    --repeat_times 1
```

If you see an error like `Environment variable 'OPENAI_API_KEY' not found`, your `.env` file isn't set up correctly.

---

## Troubleshooting

### "Environment variable 'OPENAI_API_KEY' not found"

**Solution:**
1. Make sure you created a `.env` file in the project root
2. Check that the variable name matches exactly (case-sensitive)
3. Install python-dotenv: `pip install python-dotenv`
4. Or set the environment variable manually: `export OPENAI_API_KEY="sk-..."`

### "I accidentally committed my API key!"

**Immediate actions:**
1. Rotate/regenerate the API key immediately in your provider's dashboard
2. Remove the key from git history (contact your team lead if unsure)
3. Check if the repository is public - if so, rotate keys ASAP!

**Prevention:**
1. Use `git diff` before every commit
2. Consider using git hooks (pre-commit) to scan for keys
3. Enable GitHub secret scanning if using GitHub

### "dotenv not installed"

This is optional. You can either:

**Option A:** Install it:
```bash
pip install python-dotenv
```

**Option B:** Use system environment variables instead:
```bash
export OPENAI_API_KEY="sk-..."
```

---

## Example Workflow

### Initial Setup

```bash
# 1. Install dependencies
pip install python-dotenv

# 2. Create .env from template
cp .env.example .env

# 3. Edit .env with your keys
nano .env  # or vim, code, etc.

# 4. Verify it's ignored by git
git status  # .env should NOT appear here
```

### Creating a New Model Config

```bash
# 1. Copy example config
cp configs/example-gpt-5-mini-reactive.yaml configs/my-claude-reactive.yaml

# 2. Edit to use environment variable
# Change api_key to: ${OPENAI_API_KEY}

# 3. Or just add your key directly (file is gitignored)
# This file won't be committed because of my-* pattern
```

### Running Experiments

```bash
# Your API keys are automatically loaded!
agile_eval --game freeway \
    --mode reactive \
    --reactive-model-config configs/example-gpt-5-mini-reactive.yaml \
    --settings freeway_H_8192_reactive_4096
```

---

## How It Works

The `BaseAgent` class in `src/realtimegym/agents/base.py` automatically:

1. Tries to load `.env` file using `python-dotenv` (if installed)
2. Reads your YAML config file
3. Detects `${VAR_NAME}` patterns in the `api_key` field
4. Replaces them with values from environment variables
5. Raises a clear error if the variable is not set

**Code reference:** `src/realtimegym/agents/base.py:58` (config_model1) and `src/realtimegym/agents/base.py:72` (config_model2)

---

## Questions?

- Check `.gitignore` to see what patterns are ignored
- Look at `.env.example` for required variable names
- See example configs in `configs/example-*.yaml` for reference
- All example configs now use `${VAR_NAME}` syntax safely

**Remember:** Never commit real API keys to version control!
