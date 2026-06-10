# Engineering Instructions

## Mandatory Scope

This entire file is mandatory policy and applies to every project, regardless of language, platform, framework, or domain. Nothing in this file is optional, advisory, or best-effort unless a line explicitly says otherwise.

Every agent, assistant, script, tool, and implementation plan must follow these requirements before, during, and after code changes. If any instruction conflicts with speed, convenience, assumptions, or a preferred implementation style, this file wins.

Do not reinterpret these requirements as suggestions. Do not skip them because the task seems small. Do not treat compliance as complete unless the final answer clearly shows how the relevant requirements were satisfied.

EVERYTHING IN THIS FILE IS REQUIRED.
---

## Persistent Memory

Maintain a durable, human-readable handoff document named `Persistent_Memory.md` in the project root. Its single purpose is continuity: if context is compacted, the session ends, or a new agent picks up the work, that document must contain enough detail to continue **exactly** where the previous work left off, with full knowledge of what was done and why.

This document describes the current state of the work only. It must **not** contain these engineering principles or general policy — those live in this file. `Persistent_Memory.md` holds the project's living state and handoff information, nothing else.

### Startup

1. At the start of any meaningful session, read `Persistent_Memory.md` fully before doing anything else.
2. If no persistent memory file exists, create `Persistent_Memory.md`, then ask the user what they want to happen (the goal, the task, and any constraints) before proceeding. Populate it from their answer and from what you learn about the project.

### What it must contain every single time

Keep enough detail that a fresh agent could resume without re-asking and without re-deriving prior work:

- The current goal and the specific task being worked on.
- What has been done so far, and the current state of the code/system.
- What is in progress right now, including the exact point you stopped at.
- Concrete next steps, in order.
- Open questions and hypotheses, each with its validation status (confirmed, disproven, or unverified).
- Key decisions made and the reasoning behind them.
- Relevant file paths, commands, artifacts, logs, and how to reproduce or verify.
- Known blockers, risks, or constraints.

### Maintenance (required, every session)

`Persistent_Memory.md` must always be kept current. Update it before ending a meaningful session, after making changes, after receiving new information or logs, after building or producing artifacts, and after confirming or disproving a hypothesis.

It is a living current-state document, not an append-only history. Continuously remove or rewrite anything that is outdated, untrue, superseded, or no longer serves the current purpose, so the document stays clean and easy to understand. For example, once an issue has been confirmed solved, remove the notes describing it as broken or under investigation; keep only the resolution if it still matters going forward. Stale or contradictory information is a defect — fix it as you find it.

---

## Core Principle

Produce durable, root-cause fixes with the smallest necessary change. All code must be intentional: every line must have a clear reason to exist. Do not write bloated, padded, or speculative code. Do not use workarounds, temporary mitigations, shortcuts, bypasses, feature disabling, code removal, error suppression, or "make the symptom disappear" changes unless explicitly requested.

A fix is not acceptable if it only avoids the failing path, disables the affected system, removes the problematic component, catches and ignores the error, changes behavior to avoid the crash, or narrows the issue without correcting the underlying cause. When in doubt, choose durable engineering over speed. Do not ask me to accept a workaround. Do not present a workaround as a fix.

---

## Compatibility and Portability

Changes must work for every supported platform, environment, and configuration the project targets — not only the current machine. Do not assume a specific operating system, version, vendor, hardware capability, CPU/architecture, runtime version, locale, install path, permission level, or configuration unless that requirement is explicitly verified and documented.

When implementing, fixing, or refactoring anything that interacts with the operating system, hardware, drivers, system services, filesystems, networking, or low-level/platform APIs, account for the range of environments the project supports, including both fresh and upgraded systems and unusual or restricted configurations.

Compatibility-sensitive code must:

1. Detect capabilities instead of assuming them.
2. Handle unsupported or unavailable features with clear, intentional behavior.
3. Avoid hardcoded vendor-, version-, path-, or hardware-specific assumptions unless required and justified.
4. Preserve existing support for other environments while fixing the current issue.
5. Include edge-case coverage or regression tests for materially different platforms, versions, architectures, permission states, or configuration variants when relevant.

A task is not complete if it only works on the developer's machine, one platform, one version, or one common configuration. The implementation must be robust across the project's full intended environment. For low level it must work on everyones machine and be robust and think of every edge case.

---

## Build and Environment Portability

The project must build and function consistently for every developer and intended user, not only for the current machine.

Do not hardcode local machine paths, usernames, drive letters, absolute build folders, SDK locations, tool paths, cache paths, temporary paths, or device-specific locations such as:

- `C:\Users\SomeName\...`
- `/home/someone/...`
- Desktop or Downloads folders.
- Local clone paths.
- One developer's IDE, compiler, SDK, language runtime, or package-manager install path.
- Machine-specific identifiers, device IDs, or environment-specific keys unless explicitly required and detected.

Use portable project-relative paths, documented environment variables, build-system discovery, toolchain files, configuration files, or clear user-provided settings instead.

If a dependency, path, SDK, tool, or capability is required:

1. Detect it through the build system or runtime capability checks.
2. Document the requirement clearly.
3. Fail with an actionable error if it is missing.
4. Do not silently fall back to a developer-specific path.
5. Verify the solution works from a clean checkout on a different machine path.

A change is not acceptable if it only builds or works because of files, folders, environment variables, cached artifacts, or tools that exist only on the current developer's device.

---

## Required Research and Context7 Usage

Before every code change, use the available project context and Context7. This is required, not optional.

For every change:

1. Read the relevant project files before editing.
2. Use Context7 for current library, framework, API, toolchain, and platform documentation when the change touches external APIs, dependencies, build systems, platform behavior, or anything that may have changed over time.
3. Prefer official documentation, source code, project docs, vendor docs, standards, and well-established references over guesses, blog posts, or outdated examples.
4. Verify that the planned approach matches the actual versions, APIs, configuration, and constraints used by this project.
5. State what documentation or project context was checked before changing code.

When the user asks for research, the research must be deep, thorough, credible, and practical. It must prioritize sources that are known to be authoritative and solutions that are known to work. Do not give shallow research, unsupported claims, random snippets, or guesses.

Research is required before code changes so the implementation is based on verified information, not assumptions.

---

## Before Changing Anything

Before implementing, stop and make the task clear.

1. Read and follow the project's instruction files, such as `AGENTS.md`, before making changes.
2. Read the relevant existing implementation before proposing edits.
3. Use Context7 when documentation, APIs, libraries, tools, or platform behavior are relevant.
4. State assumptions explicitly.
5. If something is unclear, name what is confusing and ask before proceeding.
6. If multiple interpretations exist, present them instead of silently choosing one.
7. If a simpler approach exists, say so.
8. Push back when the requested approach seems overcomplicated, risky, or inconsistent with the project.
9. Define success criteria before making changes.

Do not guess, assume, or speculate about the cause of a bug when the available evidence is insufficient.

Before proposing or making a fix, clearly distinguish between:

1. What is known from evidence.
2. What is suspected but not yet proven.
3. What additional diagnostics are needed to prove or disprove the suspicion.

Do not proceed with a code change as the fix unless the failure mechanism has been confirmed by evidence, reproduction, logs, tests, traces, diagnostics, or code-path analysis.

---

## Root-Cause Diagnosis

Your task is always to produce a permanent root-cause fix.

If the root cause is not yet known, your next step must be to improve observability and diagnostics until the failure mechanism can be identified with confidence.

Diagnostic work means adding targeted information that reveals the exact failing component, state, branch, input, timing, assumption, or boundary. It does not mean repeatedly changing nearby code until the symptom changes.

Useful diagnostics may include:

- Targeted debug prints with file, function, branch, state, key variable values, error codes, and timestamps.
- Assertions that prove required invariants at the point where they matter.
- Structured logs around the suspected boundary or state transition.
- Error context that includes the operation attempted, the exact failure, and relevant environment details.
- Traces that show call order, thread, timing, ownership, lifetimes, or data flow.
- Temporary diagnostic output that can be removed or reduced after the root cause is confirmed.
- Minimal reproduction checks or test instrumentation that proves the failure mechanism.

Diagnostics must be specific enough to answer:

1. Where exactly is the failure happening?
2. Which condition, value, state, or assumption is wrong?
3. Why is that wrong state possible?
4. Which code path produced it?
5. What is the smallest correct place to fix it?

Do not brute force fixes. Do not keep iterating in the general area without proof. If the current evidence does not identify the exact cause, add better diagnostics before changing logic.

Diagnostic changes are allowed only to discover the root cause. They are not a substitute for the final fix.

For every bug, crash, failure, regression, or implementation issue:

1. Identify the actual root cause and failure mechanism.
2. Fix the broken mechanism itself, not just the visible symptom.
3. Preserve the intended architecture, behavior, and feature set unless a design change is explicitly approved.
4. Search the entire codebase for equivalent instances of the same flawed pattern.
5. Fix all equivalent instances, not just the one that failed first.
6. Add or update appropriate regression protection so the same class of issue cannot be silently reintroduced.
7. Do not mark the task complete until the fix, same-pattern audit, and regression guard all pass.
8. Clearly state:

   - What root cause was found.
   - What was fixed.
   - What was audited.
   - What regression guard was added.
   - Why the issue should not recur.

Search broadly enough to include adjacent modules, shared helpers, generated or configured entry points, startup paths, tests, build scripts, and any code that can exercise the same failure mechanism.

Do not stop at the first discovered instance. Fix the entire class of problem everywhere it exists, and add enforcement so it cannot be reintroduced silently.

A task is not complete just because the current error no longer appears. It is complete only when the underlying failure mechanism has been corrected everywhere the same pattern exists, and the project has appropriate protection against reintroducing that class of bug.

---

## Regression Protection

Regression protection means the smallest reliable check that would fail if the same bug or same class of bug came back.

Do not automatically create new scripts for regression protection. A new script is only acceptable when it is the simplest maintainable option and when no existing test, build check, lint rule, assertion, validation path, CI step, or project harness can cover the issue.

Prefer regression protection in this order:

1. Existing unit, integration, or end-to-end tests.
2. Existing project test harnesses or build checks.
3. Focused test cases added to the closest relevant test file.
4. Assertions or validation checks that enforce an invariant at the correct boundary.
5. Existing lint/static-analysis/configuration rules.
6. A small new test file only when no suitable test location exists.
7. A new script only when the project has no better existing mechanism and the script is clearly justified.

Regression guards must be directly tied to the confirmed root cause. They must not be broad, unrelated, speculative, or created just to satisfy the words "regression guard."

A regression guard is acceptable only if:

1. It would have caught the confirmed bug.
2. It can be run by other developers without local machine assumptions.
3. It fits the project's existing test/build style.
4. It does not add unnecessary maintenance burden.
5. It does not hardcode user-specific paths or environment details.
6. It is documented in the final response with the exact command/check used.

If no automated regression guard is practical, explain why and provide the strongest available alternative, such as a targeted manual verification checklist, runtime assertion, build-time validation, or diagnostic check. Do not invent unnecessary scripts.

---

## Simplicity First

Write the minimum code that solves the confirmed problem. All code must be intentional and free of bloat. If a line, function, abstraction, or file does not have a clear reason to exist, it should not be there.

Do not add speculative features, abstractions, configurability, or flexibility that was not requested.

Avoid:

- Features beyond what was asked.
- Abstractions for single-use code.
- "Future-proofing" without evidence.
- Error handling for impossible scenarios.
- Large rewrites when a small fix is sufficient.
- New scripts, tools, or frameworks when an existing project mechanism is enough.

If the solution is longer or more complex than necessary, simplify it before presenting it.

Ask: "Would a senior engineer say this is overcomplicated?"

If yes, rewrite it.

---

## Surgical Changes

Touch only what is required to satisfy the request.

When editing existing code:

1. Do not improve adjacent code, comments, formatting, or structure unless directly required.
2. Do not refactor unrelated code.
3. Match the existing project style, even if you would normally do it differently.
4. If you notice unrelated dead code, mention it instead of deleting it.
5. Remove imports, variables, functions, or files that your own changes made unused.
6. Do not remove pre-existing dead code unless explicitly asked.

Every changed line should trace directly to the user's request, the confirmed root cause, the same-pattern audit, or the regression guard.

---

## Comments and Documentation

Comments must help future readers understand code that is not obvious from the implementation itself. Comments must not replace reading the code.

Write comments that are short, concise, and specific. Include detail only when it prevents misunderstanding or documents a non-obvious constraint.

When adding or updating comments that describe behavior, assumptions, platform details, external APIs, offsets, compatibility constraints, or implementation reasoning, include the date in `YYYY-MM-DD` format so future readers know when the statement was written or verified.

Example:

```
// 2026-06-05: This API returns an "insufficient buffer" error on the first call when it is used only to query the required size.
```

Do not trust old comments over code. If a comment conflicts with the implementation, tests, logs, or current documentation, treat the comment as stale and verify the behavior from the source of truth before acting on it.

When changing code near an outdated or misleading comment:

1. Update or remove the stale comment if it is directly related to the change.
2. Do not preserve comments that are known to be wrong.
3. Do not add broad comments that will quickly become outdated.
4. Prefer comments that explain why a non-obvious decision exists, not comments that repeat what the code already says.

---

## Goal-Driven Execution

Turn tasks into verifiable goals.

Examples:

- "Add validation" means: write or update tests for invalid inputs, then make them pass.
- "Fix the bug" means: reproduce the bug with a test, diagnostic evidence, logs, traces, or code-path proof, then make the check pass.
- "Refactor X" means: verify behavior before and after the refactor.

For multi-step tasks, state a brief plan before implementation:

1. Step → verify with specific check.
2. Step → verify with specific check.
3. Step → verify with specific check.

Success criteria must be concrete. Avoid vague goals like "make it work."

Loop until the defined checks pass.

---

## Completion Standard

Do not mark work complete until all of the following are true:

1. The root cause is identified with evidence.
2. The broken mechanism is fixed directly.
3. Equivalent instances of the same flawed pattern have been searched for and addressed.
4. Appropriate regression protection has been added or updated without unnecessary scripts.
5. Relevant tests, checks, or builds pass.
6. Build and runtime assumptions are portable and do not depend on one developer's machine.
7. Context7 and relevant project documentation were checked when applicable.
8. `Persistent_Memory.md` has been updated to reflect the current state, and outdated or resolved information has been pruned.
9. The final response explains:

   - The confirmed root cause.
   - The fix made.
   - The scope of the audit.
   - The regression guard or why a different verification method was used.
   - The verification performed, including exact commands/checks used.
   - The documentation, project context, and Context7 references checked when applicable.

The goal is fewer unnecessary diffs, fewer overcomplicated rewrites, fewer machine-specific changes, better diagnosis before implementation, and fixes that are correct for the whole project and user base.
