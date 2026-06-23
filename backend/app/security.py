from __future__ import annotations

import secrets
from collections.abc import Sequence

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


PUBLIC_PATHS = ("/health", "/docs", "/redoc", "/openapi.json")


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        api_key: str = "",
        public_paths: Sequence[str] = PUBLIC_PATHS,
    ):
        super().__init__(app)
        self.api_key = api_key
        self.public_paths = tuple(public_paths)

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.api_key or request.method == "OPTIONS" or self._is_public(request.url.path):
            return await call_next(request)

        provided = request.headers.get("x-api-key", "")
        authorization = request.headers.get("authorization", "")
        if not provided and authorization.lower().startswith("bearer "):
            provided = authorization[7:].strip()

        if not secrets.compare_digest(provided, self.api_key):
            return JSONResponse(
                {"detail": "Cle API manquante ou invalide."},
                status_code=401,
            )
        return await call_next(request)

    def _is_public(self, path: str) -> bool:
        return any(path == public_path for public_path in self.public_paths)
