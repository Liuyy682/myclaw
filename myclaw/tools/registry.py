from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from myclaw.providers.base import ToolCallRequest
from myclaw.config import DEFAULT_MAX_TOOL_RESULT_CHARS, DEFAULT_TOOL_TIMEOUT_SECONDS, TOOL_RESULT_TRUNCATED_TEMPLATE
from myclaw.tools.base import Tool, ToolRuntimeContext, tool_context
from myclaw.observability import SpanHandle, current_observability
from myclaw.tools.security import PolicyGate, ResultEnvelope, SecurityStore, canonical_args_hash, operation_id, result_digest


class ToolRegistry:
    """Registry for OpenAI-style function tools."""

    def __init__(
        self,
        security_store: SecurityStore | Path | str | None = None,
        *,
        security_path: Path | str | None = None,
        security_db_path: Path | str | None = None,
        workspace: Path | str | None = None,
        policy_gate: PolicyGate | None = None,
        tool_timeout_seconds: float | None = DEFAULT_TOOL_TIMEOUT_SECONDS,
        timeout_seconds: float | None = None,
        max_result_chars: int | None = DEFAULT_MAX_TOOL_RESULT_CHARS,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._cached_definitions: list[dict[str, Any]] | None = None
        if isinstance(security_store, (str, Path)):
            security_path = security_store
            security_store = None
        security_path = security_path or security_db_path
        if timeout_seconds is not None:
            tool_timeout_seconds = timeout_seconds
        if security_store is None and (security_path is not None or workspace is not None):
            if security_path is not None:
                security_store = SecurityStore(path=security_path)
            else:
                security_store = SecurityStore(workspace=workspace)
        self.security_store = security_store
        self.policy_gate = policy_gate if policy_gate is not None else PolicyGate(security_store)
        self.tool_timeout_seconds = tool_timeout_seconds
        self.max_result_chars = max_result_chars
        self._exclusive_lock: asyncio.Lock | None = None

    def set_security_store(self, security_store: SecurityStore | Path | str) -> None:
        """Install the workspace security store on an existing registry."""

        if isinstance(security_store, (str, Path)):
            security_store = SecurityStore(path=security_store)
        self.security_store = security_store
        if getattr(self.policy_gate, "store", None) is None:
            self.policy_gate = PolicyGate(security_store)

    set_mcp_security_store = set_security_store

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._cached_definitions = None

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._cached_definitions = None

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def definitions(self) -> list[dict[str, Any]]:
        if self._cached_definitions is not None:
            return self._cached_definitions

        self._cached_definitions = [
            self._tool_schema(tool)
            for tool in sorted(self._tools.values(), key=lambda candidate: candidate.name)
        ]
        return self._cached_definitions

    def prepare_call(self, request: ToolCallRequest) -> tuple[Tool | None, dict[str, Any], str | None]:
        tool = self._tools.get(request.name)
        if tool is None:
            return None, {}, f"Error: Tool '{request.name}' not found. Available: {', '.join(self.tool_names)}"
        if not isinstance(request.arguments, dict):
            return None, {}, (
                f"Error: Tool '{request.name}' arguments must be a JSON object, "
                f"got {type(request.arguments).__name__}"
            )
        arguments = dict(request.arguments)
        cast_params = getattr(tool, "cast_params", None)
        if callable(cast_params):
            try:
                cast_arguments = cast_params(arguments)
            except Exception as exc:
                return None, {}, f"Error casting {request.name}: {exc}"
            if cast_arguments is None:
                cast_arguments = arguments
            arguments = cast_arguments
            if not isinstance(arguments, dict):
                return None, {}, f"Error casting {request.name}: cast_params must return a dict"
        input_model = getattr(tool, "input_model", None)
        if input_model is not None:
            try:
                validated = input_model.model_validate(arguments)
            except ValidationError as exc:
                return None, {}, f"Error validating {request.name}: {_format_validation_error(exc)}"
            arguments = validated.model_dump(exclude_unset=True)
        validate_params = getattr(tool, "validate_params", None)
        if callable(validate_params):
            try:
                validate_params(arguments)
            except ValidationError as exc:
                return None, {}, f"Error validating {request.name}: {_format_validation_error(exc)}"
            except Exception as exc:
                return None, {}, f"Error validating {request.name}: {exc}"
        return tool, arguments, None

    async def execute(
        self,
        request: ToolCallRequest,
        *,
        max_result_chars: int | None = None,
        context: ToolRuntimeContext | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        envelope = await self.execute_envelope(
            request,
            max_result_chars=max_result_chars,
            context=context,
            timeout_seconds=timeout_seconds,
        )
        if envelope["status"] == "ok":
            return str(envelope["result"])
        return str(envelope["error"])

    async def execute_envelope(
        self,
        request: ToolCallRequest,
        *,
        max_result_chars: int | None = None,
        context: ToolRuntimeContext | None = None,
        timeout_seconds: float | None = None,
    ) -> ResultEnvelope:
        """Authorize and execute a tool, returning a structured internal receipt.

        ``execute`` intentionally remains the historical string API for model
        messages.  Callers that need security status, operation identity, or
        result metadata should use this envelope instead.
        """

        runtime_context = self._runtime_context(context)
        result_limit = self._result_limit(max_result_chars)
        observability = current_observability()
        scope = (
            observability.span(
                "tool.execute",
                "tool",
                attributes={
                    "tool_name": request.name,
                    "argument_keys": sorted(request.arguments) if isinstance(request.arguments, dict) else [],
                },
            )
            if observability is not None
            else contextlib.nullcontext(SpanHandle())
        )
        with scope as span:
            tool, arguments, error = self.prepare_call(request)
            if error is not None:
                span.set_error(error, error_type="ToolPreparationError")
                message = self._truncate_result(error, result_limit)
                return self._envelope(
                    request,
                    status="error",
                    result=message,
                    error=message,
                    error_type="ToolPreparationError",
                )

            assert tool is not None
            args_hash = canonical_args_hash(arguments)
            local_read = PolicyGate.is_explicit_local_read(tool)
            operation = None if local_read or request.name == "ask_user" else operation_id(
                runtime_context.session_key,
                request.id,
                request.name,
                args_hash,
            )
            store = self.security_store
            if store is not None and operation is not None and store.get_operation(operation) is not None:
                message = f"Error: duplicate operation for tool '{request.name}'"
                span.set_error(message, error_type="DuplicateOperationError")
                self._audit(
                    request,
                    runtime_context,
                    decision="deny",
                    args_hash=args_hash,
                    operation_id=operation,
                    arguments=arguments,
                    error_type="DuplicateOperationError",
                )
                return self._envelope(
                    request,
                    status="duplicate",
                    result=message,
                    error=message,
                    error_type="DuplicateOperationError",
                    operation_id=operation,
                    args_hash=args_hash,
                    decision="deny",
                )

            authorization = self.policy_gate.authorize(
                tool,
                arguments,
                context=runtime_context,
                tool_call_id=request.id,
            )
            decision = await authorization if inspect.isawaitable(authorization) else authorization
            if decision.action != "allow":
                message = f"Error: Tool '{request.name}' denied by security policy: {decision.reason}"
                span.set_error(message, error_type="ToolPolicyError")
                self._audit(
                    request,
                    runtime_context,
                    decision=decision.action,
                    args_hash=args_hash,
                    operation_id=operation,
                    arguments=arguments,
                    error_type="ToolPolicyError",
                )
                return self._envelope(
                    request,
                    status="denied" if decision.action == "deny" else "approval_required",
                    result=message,
                    error=message,
                    error_type="ToolPolicyError",
                    operation_id=operation,
                    args_hash=args_hash,
                    decision=decision.action,
                )

            # Approval without a durable ledger would make a side effect
            # replayable after a process restart.  Explicit local reads and the
            # interactive ask_user primitive are the only no-store exceptions.
            if store is None and not local_read and request.name != "ask_user":
                message = f"Error: Tool '{request.name}' requires the security store"
                span.set_error(message, error_type="SecurityStoreRequiredError")
                self._audit(
                    request,
                    runtime_context,
                    decision="deny",
                    args_hash=args_hash,
                    operation_id=operation,
                    arguments=arguments,
                    error_type="SecurityStoreRequiredError",
                )
                return self._envelope(
                    request,
                    status="denied",
                    result=message,
                    error=message,
                    error_type="SecurityStoreRequiredError",
                    operation_id=operation,
                    args_hash=args_hash,
                    decision="deny",
                )

            if store is not None and operation is not None:
                subject = self._security_subject(runtime_context)
                reserved = store.prepare_operation(
                    operation_id=operation,
                    security_subject=subject,
                    session_key=runtime_context.session_key,
                    channel=runtime_context.channel,
                    tool_call_id=request.id,
                    tool=request.name,
                    args_hash=args_hash,
                )
                if not reserved:
                    message = f"Error: duplicate operation for tool '{request.name}'"
                    span.set_error(message, error_type="DuplicateOperationError")
                    self._audit(
                        request,
                        runtime_context,
                        decision="deny",
                        args_hash=args_hash,
                        operation_id=operation,
                        arguments=arguments,
                        error_type="DuplicateOperationError",
                    )
                    return self._envelope(
                        request,
                        status="duplicate",
                        result=message,
                        error=message,
                        error_type="DuplicateOperationError",
                        operation_id=operation,
                        args_hash=args_hash,
                        decision="deny",
                    )
                store.start_operation(operation)

            try:
                if bool(getattr(tool, "exclusive", False)):
                    if self._exclusive_lock is None:
                        self._exclusive_lock = asyncio.Lock()
                    async with self._exclusive_lock:
                        result = await self._invoke(tool, arguments, runtime_context, timeout_seconds)
                else:
                    result = await self._invoke(tool, arguments, runtime_context, timeout_seconds)
            except asyncio.TimeoutError as exc:
                message = self._timeout_message(request)
                span.set_error(exc, error_type="ToolTimeoutError")
                if store is not None and operation is not None:
                    store.finish_operation(operation, status="unknown", error_type="ToolTimeoutError")
                self._audit(
                    request,
                    runtime_context,
                    decision="allow",
                    args_hash=args_hash,
                    operation_id=operation,
                    arguments=arguments,
                    error_type="ToolTimeoutError",
                )
                return self._envelope(
                    request,
                    status="timeout",
                    result=message,
                    error=message,
                    error_type="ToolTimeoutError",
                    operation_id=operation,
                    args_hash=args_hash,
                    decision="allow",
                )
            except asyncio.CancelledError:
                if store is not None and operation is not None:
                    store.finish_operation(operation, status="unknown", error_type="CancelledError")
                self._audit(
                    request,
                    runtime_context,
                    decision="allow",
                    args_hash=args_hash,
                    operation_id=operation,
                    arguments=arguments,
                    error_type="CancelledError",
                )
                raise
            except Exception as exc:
                message = f"Error executing {request.name}: {exc}"
                span.set_error(exc, error_type=type(exc).__name__)
                if store is not None and operation is not None:
                    store.finish_operation(operation, status="failed", error_type=type(exc).__name__)
                self._audit(
                    request,
                    runtime_context,
                    decision="allow",
                    args_hash=args_hash,
                    operation_id=operation,
                    arguments=arguments,
                    error_type=type(exc).__name__,
                )
                return self._envelope(
                    request,
                    status="error",
                    result=self._truncate_result(message, result_limit),
                    error=self._truncate_result(message, result_limit),
                    error_type=type(exc).__name__,
                    operation_id=operation,
                    args_hash=args_hash,
                    decision="allow",
                )

            degraded_error_type = (
                "DegradedExecSandbox"
                if request.name == "exec" and isinstance(result, dict) and result.get("degraded") is True
                else None
            )
            normalized = self._normalize_result(result)
            returned = self._truncate_result(normalized, result_limit)
            span.set_attribute("result_chars", len(returned))
            result_hash, result_len = result_digest(returned)
            if store is not None and operation is not None:
                store.finish_operation(
                    operation,
                    status="succeeded",
                    result_hash=result_hash,
                    result_len=result_len,
                )
            self._audit(
                request,
                runtime_context,
                decision="allow",
                args_hash=args_hash,
                operation_id=operation,
                arguments=arguments,
                result_hash=result_hash,
                result_len=result_len,
                error_type=degraded_error_type,
            )
            return self._envelope(
                request,
                status="ok",
                result=returned,
                operation_id=operation,
                args_hash=args_hash,
                decision="allow",
                result_hash=result_hash,
                result_len=result_len,
            )

    async def _invoke(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        runtime_context: ToolRuntimeContext,
        timeout_seconds: float | None = None,
    ) -> Any:
        set_context = getattr(tool, "set_context", None)
        if callable(set_context):
            set_context(runtime_context)

        async def invoke() -> Any:
            with tool_context(runtime_context):
                return await tool.execute(**arguments)

        effective_timeout = self.tool_timeout_seconds if timeout_seconds is None else timeout_seconds
        if effective_timeout is None:
            return await invoke()
        timeout = max(0.001, float(effective_timeout))
        return await asyncio.wait_for(invoke(), timeout=timeout)

    def _result_limit(self, requested: int | None) -> int | None:
        configured = self.max_result_chars
        if requested is None:
            return configured
        if configured is None:
            return max(0, int(requested))
        return max(0, min(int(requested), int(configured)))

    def _timeout_message(self, request: ToolCallRequest) -> str:
        if self.tool_timeout_seconds is None:
            return f"Error executing {request.name}: timed out"
        return f"Error: Tool '{request.name}' timed out after {self.tool_timeout_seconds:g} seconds"

    @staticmethod
    def _security_subject(context: ToolRuntimeContext) -> str:
        return str(
            context.security_subject
            or context.subject
            or context.metadata.get("security_subject", "")
            or ""
        )

    def _audit(
        self,
        request: ToolCallRequest,
        context: ToolRuntimeContext,
        *,
        decision: str,
        args_hash: str,
        operation_id: str | None,
        arguments: dict[str, Any],
        result_hash: str | None = None,
        result_len: int | None = None,
        error_type: str | None = None,
    ) -> None:
        if self.security_store is None:
            return
        self.security_store.record_audit_event(
            security_subject=self._security_subject(context),
            session_key=context.session_key,
            channel=context.channel,
            tool_call_id=request.id,
            tool=request.name,
            decision=decision,
            operation_id=operation_id,
            argument_keys=list(arguments.keys()),
            args_hash=args_hash,
            result_hash=result_hash,
            result_len=result_len,
            error_type=error_type,
        )

    @staticmethod
    def _envelope(
        request: ToolCallRequest,
        *,
        status: str,
        result: str,
        error: str | None = None,
        error_type: str | None = None,
        operation_id: str | None = None,
        args_hash: str | None = None,
        decision: str | None = None,
        result_hash: str | None = None,
        result_len: int | None = None,
    ) -> ResultEnvelope:
        return {
            "status": status,
            "ok": status == "ok",
            "untrusted": True,
            "tool": request.name,
            "tool_call_id": request.id,
            "operation_id": operation_id,
            "args_hash": args_hash,
            "decision": decision,
            "result": result,
            "content": result,
            "error": error,
            "error_type": error_type,
            "result_hash": result_hash,
            "result_len": result_len,
        }

    @staticmethod
    def _tool_schema(tool: Tool) -> dict[str, Any]:
        to_schema = getattr(tool, "to_schema", None)
        if callable(to_schema):
            schema = to_schema()
            if isinstance(schema, dict):
                return schema
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def _runtime_context(self, context: ToolRuntimeContext | None) -> ToolRuntimeContext:
        runtime_context = context or ToolRuntimeContext()
        if not runtime_context.tool_names:
            runtime_context.tool_names = sorted(self.tool_names)
        return runtime_context

    @staticmethod
    def _normalize_result(result: Any) -> str:
        if result is None:
            return "(empty)"
        if isinstance(result, str):
            return result if result else "(empty)"
        try:
            return json.dumps(result, ensure_ascii=False)
        except TypeError:
            return str(result) if str(result) else "(empty)"

    @staticmethod
    def _truncate_result(result: str, max_result_chars: int | None) -> str:
        if max_result_chars is None or len(result) <= max_result_chars:
            return result
        omitted = len(result) - max_result_chars
        return f"{result[:max_result_chars]}\n{TOOL_RESULT_TRUNCATED_TEMPLATE.format(omitted=omitted)}"

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def _format_validation_error(error: ValidationError) -> str:
    """Return one redacted, stable first-error summary for tool callers."""

    details = error.errors(include_url=False, include_input=False)
    if not details:
        return "input: Invalid value"
    first = details[0]
    location = first.get("loc") or ("input",)
    field = ".".join(str(part) for part in location)
    message = " ".join(str(first.get("msg", "Invalid value")).split())
    return f"{field}: {message}"
