# Action Label Prompt

Use this with a video VLM after SAM 3 has produced tracks/objects.

```text
You are labeling restaurant surveillance footage for training an action model.

Return strict JSON. Do not include prose.

Labels:
- waiting_for_service
- ordering
- eating
- drinking
- food_delivered
- bussing
- leaving
- needs_attention
- dirty_table
- no_relevant_action

For each visible table/person group, return:
- label
- start_time
- end_time
- confidence from 0.0 to 1.0
- evidence: short visual reason
- uncertain: true/false

Only use a label when there is visible evidence across multiple frames.
If the clip is ambiguous, use no_relevant_action or uncertain=true.
```
