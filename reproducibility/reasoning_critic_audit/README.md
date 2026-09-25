# Reasoning and critic audit

Script: `verify_audit.py`

Command:

```bash
python reproducibility/reasoning_critic_audit/verify_audit.py
```

Validated current results include 3000/3000 coverage for both ReasoningAgent and MedicalCriticAgent, 57000/57000 reasoning-grounding checks, 45000/45000 supported critique checks, zero unsupported critique claims, and 3000/3000 non-interference for both reasoning and critique.

These are implementation and architectural-integrity checks. They do not establish clinical reasoning quality, clinical correctness, expert agreement, explanation usefulness, or downstream utility.
