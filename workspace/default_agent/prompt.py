AGENT_ACT_PROMPT = """
==== Environment Instruction ====
{env_instruction}

==== Action Space ====
{action_space}

==== Output Format ====

==== thinking output format ====
Before outputting an action, you should think step by step, and write any necessary reasoning (such as environment rules or information relevant for future actions) inside the <thinking_memory></thinking_memory> tag.

==== Action Output Format ====
When you output the action, 
you should output the action name and parameters in the format python dict can parse, and warpped it in <action></action> tag, and only one action.
Such as, 
<action>
{{
    "action": "",
    "params": {{
        "<param_name>": "<param_value>"
    }}
}}
</action>

The thinking and action should be outputted separately:
- First, write your reasoning inside <thinking_memory></thinking_memory> tag
- Then, output the action inside <action></action> tag, and the action content should can be parsed by python dict.

==== Past Actions ====
Your recent actions are:
{recent_actions}

==== Now, your observation is:====
{obs}
"""


LEARNED_INSTRUCTION_PROMPT = None