## Reasoning resource allocation

Use High as the default reasoning level for this project. Preserve scientific reliability while controlling quota consumption.

- Medium: mechanical, reversible work such as paths, routine configuration, parsing, plotting, formatting, and summaries.
- High: normal project management and engineering, including paper reading, implementation, testing, log inspection, parameter checks, and result validation.
- Extra High: ambiguous paper details, cross-module architecture, unexplained numerical disagreement, and consequential technical decisions.
- Ultra: only a bounded core problem that remains unresolved after evidence-driven diagnosis at lower levels.

Before recommending a higher level, report:

1. the single blocker;
2. the minimal evidence demonstrating it;
3. causes already tested and excluded;
4. the exact decision required from the higher-effort run;
5. the acceptance test.

Do not claim that the active reasoning level changed unless the runtime confirms it. If effort-specific subagents are supported, delegate only the bounded diagnostic question. Return implementation, testing, monitoring, and integration to High after the decision is made.
