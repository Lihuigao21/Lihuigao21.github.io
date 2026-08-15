## Reasoning resource allocation

Use High as the default reasoning level for this project. Preserve scientific reliability while controlling quota consumption.

- Medium: mechanical, reversible work such as paths, routine configuration, parsing, plotting, formatting, and summaries.
- High: normal project management and engineering, including paper reading, implementation, testing, log inspection, parameter checks, and result validation.
- Extra High: ambiguous paper details, cross-module architecture, unexplained numerical disagreement, and consequential technical decisions.
- Ultra: proactive subagent orchestration for a large task with genuinely independent workstreams; do not treat it merely as a higher reasoning-effort setting.

Before recommending a higher level, report:

1. the single blocker;
2. the minimal evidence demonstrating it;
3. causes already tested and excluded;
4. the exact decision required from the higher-effort run;
5. the acceptance test.

Do not claim that the active reasoning level, orchestration mode, or service tier changed unless the runtime confirms it. At High, explicitly delegate independent scopes when parallel work would improve speed or review quality. Give each subagent separate file ownership and a testable output contract. Keep credentialed access, risk-stop decisions, shared-state integration, and final approval under one accountable main agent. Use Ultra when proactive multi-agent decomposition is materially useful, and use Fast only as a latency service tier rather than as a substitute for reasoning or validation.
