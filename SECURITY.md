# Security

Eluate parses untrusted input: audio files (through libsndfile and its codecs) and JSON
(saved plans, reports, the desktop shell's engine messages). A file that crashes the engine,
makes it read or write outside the paths it was given, or runs code is a security bug.

Report it privately through GitHub: **Security → Report a vulnerability** on this repository
(a private security advisory). Please do not open a public issue for it. Include the input
that triggers it if you can share it, the Eluate version and what happened.

Only the latest release is supported.
