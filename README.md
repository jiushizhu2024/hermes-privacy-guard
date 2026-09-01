# Hermes Privacy Guard Plugin

On-device PII detection and redaction plugin for Hermes Agent.

## Features

- **Rule-based detection**: Regex patterns for structured PII (email, phone, IBAN, credit cards, IPs, Chinese IDs)
- **Chinese NER**: spaCy `zh_core_web_sm` model for unstructured Chinese PII (names, addresses)
- **Optional ML model**: Ai4Privacy transformer for multilingual PII detection
- **Consistent placeholders**: Same input → same placeholder (for context consistency)

## Installation

```bash
# Clone the plugin
mkdir -p ~/.hermes/plugins && cd ~/.hermes/plugins
git clone https://github.com/jiushizhu2024/hermes-privacy-guard.git privacy-guard

# Install dependencies
pip install spacy zh-core-web-sm ai4privacy --break-system-packages

# Enable the plugin
hermes plugins enable privacy-guard
```

## Usage

The plugin automatically hooks into all Hermes tool outputs:

```python
# All tool results are automatically redacted
result = terminal("whoami")  # PII in output is redacted before reaching LLM
result = web_search("user@example.com")  # Email is redacted
```

## Supported PII Types

| Type | Pattern | Placeholder |
|------|---------|-------------|
| EMAIL | `user@example.com` | `EMAIL#<hash>` |
| PHONE | `+86-138-1234-5678`, `13812345678` | `PHONE#<hash>` |
| CREDIT_CARD | `4111111111111111` | `CC#<hash>` |
| IBAN | `DE44500105170445678901` | `IBAN#<hash>` |
| IP_ADDRESS | `192.168.1.100` | `IP#<hash>` |
| SSN (Chinese ID) | `110101199003074333` | `SSN#<hash>` |
| NAME (Chinese) | `张三` | `NAME#<hash>` |
| ADDRESS (Chinese) | `北京市朝阳区建国路100号` | `ADDR#<hash>` |

## Architecture

### Three-Layer Detection Pipeline

1. **Chinese NER** (spaCy `zh_core_web_sm`)
   - PERSON entities → NAME placeholders
   - GPE/FAC/ORG + address markers → ADDRESS placeholders
   - Smart filtering: skips pure numbers, already-redacted text, short entities

2. **Regex-based Detection**
   - EMAIL, PHONE, CREDIT_CARD, IBAN, SSN, IP_ADDRESS
   - Order matters: longer/more specific patterns first

3. **ML Model** (optional, Ai4Privacy)
   - Handles edge cases
   - Supports multilingual PII

### Placeholders

All redacted values use consistent placeholders:
- Format: `{TYPE}#{short_hash}`
- Same input → same placeholder (ensures AI tools work correctly)
- Short hash (6 chars) keeps output readable

Example:
```
Input:  "张三的邮箱是zhangsan@example.com，电话138-1234-5678"
Output: "NAME#1d841b的邮箱是EMAIL#55370d，电话PHONE#b89dd0"
```

## Configuration

### Disable ML model (regex-only mode)
```python
PrivacyGuard(use_model=False)
```

### Customize placeholders
Edit `PII_PLACEHOLDERS` dict in `__init__.py`:
```python
PII_PLACEHOLDERS = {
    "EMAIL": "EMAIL#{}",
    "PHONE": "PHONE#{}",
    # ... etc
}
```

## Testing

```bash
# Run tests
python3 /tmp/test_privacy_final.py

# Test Chinese NER
python3 /tmp/debug_ner.py

# Full validation
python3 /tmp/test_privacy_final.py
```

## Files

- `__init__.py` - Main plugin code with `PrivacyGuard` class and hook registration
- `plugin.yaml` - Plugin manifest defining hooks and metadata
- `README.md` - This file

## Dependencies

- Python 3.12+
- `spacy` - NLP framework
- `zh-core-web-sm` - Chinese language model (65MB)
- `ai4privacy` - Optional ML-based PII detection

## License

MIT License

## Contributing

Pull requests welcome! Please ensure tests pass before submitting.

---

Built with ❤️ for Hermes Agent
