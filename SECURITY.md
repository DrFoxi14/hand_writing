# Security and privacy

- Keep the API bound to `127.0.0.1` unless authentication, TLS, access control, and network threat review have been added.
- Treat handwriting photos, crops, transcription, and style embeddings as sensitive personal data.
- Do not upload personal samples or checkpoints to external services without explicit authorization.
- Do not add private data or model weights to Git. Check `.gitignore` before every commit.
- Only process handwriting the operator owns or is authorized to use. Do not use this tool to impersonate another person or produce misleading signatures, official records, or evidence.
- Report vulnerabilities privately to the repository maintainer rather than opening a public issue with exploit details.
