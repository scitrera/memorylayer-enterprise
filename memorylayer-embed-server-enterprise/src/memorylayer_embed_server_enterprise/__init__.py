# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only

"""Enterprise plugin overlay for memorylayer-embed-server.

Adds the Qwen3.5 visual-tokenizer service + ``/v1/visual-tokenize`` API
routes. The OSS embed-server's ``register_package_plugins`` loop
discovers everything in this package automatically when installed.
"""

__version__ = "0.0.1"
