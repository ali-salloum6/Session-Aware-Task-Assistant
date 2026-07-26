# Memory write policy

Long-term memory is written deliberately: on explicit user request via the
`remember` tool, or after a task via a single reflection pass that may propose
0–3 durable facts (preferences, constraints, decisions) and must abstain when
nothing lasting was learned. Near-duplicates are skipped by cosine distance so
repeated preferences do not multiply. Ephemeral arithmetic and tool noise are
not stored; empty recall must be answered as unknown, never confabulated.
