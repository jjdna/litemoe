# Security Policy

This is a public open-source project. **Do not commit sensitive information.**

## Prohibited Content

The following must **never** appear in this repository (including git history):

- ❌ IP addresses (internal or external)
- ❌ Hostnames that reveal internal infrastructure
- ❌ API keys, tokens, passwords, or secrets
- ❌ Personal identifying information (names, emails, phone numbers)
- ❌ File system paths that reveal user names or structures
- ❌ Private SSH keys or certificates

## What to Do If You Find a Leak

1. **Immediately rewrite history** using `git filter-branch` or `git filter-repo`
2. **Force push** the cleaned history to GitHub
3. **Rotate** any exposed credentials
4. Open an issue to document the incident

## Allowed

- `0.0.0.0` as a bind address placeholder (standard in server configs)
- Placeholder paths like `/path/to/model.gguf`
- Example domains like `example.com`
