# Hermes Privacy Guard Plugin

**Bidirectional on-device PII protection for Hermes Agent.**

Privacy Guard detects and hides personal data — emails, credit cards and IBANs,
credentials, phone numbers, national IDs, addresses, and names — in everything
Hermes tools return (terminal output, browser content, file reads, MCP results,
…) before it leaves the device for the AI provider. When the model replies,
placeholders are restored back to the original values, so PII always displays
correctly locally while only redacted copies leave the device.

## How It Works

```
Tool output (raw PII)                    Model reply (placeholders)
       │                                         │
       ▼                                         ▼
┌─────────────────┐                    ┌──────────────────┐
│  transform_tool  │                    │ transform_llm_    │
│  _result hook    │                    │ output hook      │
│  (OUTBOUND)      │                    │ (INBOUND)         │
│                  │                    │                   │
│  PII → placeholder│   ──── network ──→│  placeholder → PII│
│  + store mapping  │                    │  + lookup mapping │
└─────────────────┘                    └──────────────────┘
       │                                         │
       ▼                                         ▼
  Redacted copy sent                    Original PII shown
  to model                               to user locally
```

## Features

- **Bidirectional**: redacts outbound, restores inbound — PII displays correctly locally
- **Rule-based detection**: regex for email, phone, IBAN, credit cards, IPs, Chinese IDs
- **Chinese NER**: spaCy `zh_core_web_sm` for unstructured Chinese names and addresses
- **Optional ML model**: Ai4Privacy transformer for multilingual PII
- **Consistent placeholders**: same input → same placeholder (for context consistency)
- **Persistent mapping**: placeholder↔value mapping survives plugin reloads
- **Safe restore**: unknown/hallucinated placeholders are never restored

## Installation

```bash
# Clone the plugin
mkdir -p ~/.hermes/plugins && cd ~/.hermes/plugins
git clone https://github.com/jiushizhu2024/hermes-privacy-guard.git privacy-guard

# Install dependencies
pip install spacy spacy-pkuseg
# Download Chinese model (choose one):
pip install https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.8.0/zh_core_web_sm-3.8.0-py3-none-any.whl
# OR if GitHub is blocked, copy from another Python installation:
# cp -r /usr/lib/python3.12/site-packages/zh_core_web_sm /your/venv/site-packages/

# Optional: AI4Privacy ML model
pip install ai4privacy

# Enable the plugin
hermes plugins enable privacy-guard
```

## Supported PII Types

| Type | Example | Placeholder |
|------|---------|-------------|
| EMAIL | `user@example.com` | `EMAIL#<hash>` |
| PHONE | `+86-138-1234-5678` | `PHONE#<hash>` |
| CREDIT_CARD | `4111111111111111` | `CC#<hash>` |
| IBAN | `DE44500105170445678901` | `IBAN#<hash>` |
| IP_ADDRESS | `192.168.1.100` | `IP#<hash>` |
| SSN (Chinese ID) | `110101199003074333` | `SSN#<hash>` |
| NAME (Chinese) | `张三` | `NAME#<hash>` |
| ADDRESS (Chinese) | `北京市朝阳区` | `ADDR#<hash>` |

## Architecture

### Three-Layer Detection Pipeline

1. **Chinese NER** (spaCy `zh_core_web_sm`)
   - PERSON entities → NAME placeholders
   - GPE/FAC/ORG + address markers → ADDRESS placeholders
   - Idempotent: skips text already containing placeholders

2. **Regex-based Detection**
   - EMAIL, PHONE, CREDIT_CARD, IBAN, SSN, IP_ADDRESS
   - CJK-safe boundaries (no `\b`, uses ASCII-anchored lookarounds)

3. **ML Model** (optional, Ai4Privacy)
   - Handles edge cases, multilingual PII

### Bidirectional Mapping

- **Outbound** (`transform_tool_result`): PII → placeholder + store mapping
- **Inbound** (`transform_llm_output`): placeholder → original PII (from mapping)
- Mapping persisted to `~/.hermes/plugin-data/privacy-guard/mapping.json`
- Unknown placeholders are never restored (no injection risk)

## Configuration

### Disable ML model (regex + NER only)
```python
PrivacyGuard(use_model=False)
```

## Testing

```bash
# Full round-trip test
python3 /tmp/truth2.py

# NER logic test (without spaCy)
python3 /tmp/test_ner_logic.py
```

## Dependencies

- Python 3.11+
- `spacy` — NLP framework
- `spacy-pkuseg` — Chinese tokenizer
- `zh_core_web_sm` — Chinese language model (~50MB)
- `ai4privacy` — Optional ML-based PII detection

## Files

- `__init__.py` — Main plugin code with `PrivacyGuard` class and hook registration
- `plugin.yaml` — Plugin manifest defining hooks and metadata
- `requirements.txt` — Python dependencies
- `pyproject.toml` — Project configuration

## License

MIT License

## Contributing

Pull requests welcome! Please ensure tests pass before submitting.

---

Built for Hermes Agent
