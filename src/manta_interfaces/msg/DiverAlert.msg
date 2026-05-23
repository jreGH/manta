# The compressed push message delivered to the focused actor.
# Designed to fit within a severe bandwidth budget (e.g. 160 chars).

builtin_interfaces/Time stamp
uint8 priority        # 0=info  1=warning  2=safety_critical
string text           # the human-readable message
string actor_id       # primary actor that triggered this alert (may be empty)
float32 budget_fraction  # fraction of bandwidth budget consumed [0.0–1.0]
