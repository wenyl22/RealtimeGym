import re
def extract_text(text, default_value=""):
    """
    Extracts the text{...} from the input string.
    Returns the last match found or the default value if no match is found.
    """
    matches = re.findall(r'ext{(.*?)}', text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip() if text else default_value


def extract_boxed(text, default_value=""):
    """
    Extracts the \boxed{...} text from the input string.
    """
    pattern = r'oxed{' 
    start_index = text.rfind(pattern)
    if start_index == -1:
        # Try to extract content enclosed in triple backticks if \boxed{...} is not found
        triple_backtick_pattern = r"```(.*?)```"
        matches = re.findall(triple_backtick_pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return default_value
    start_index += len(pattern) - 1
    stack = []
    for i in range(start_index, len(text)):
        if text[i] == '{':
            stack.append('{')
        elif text[i] == '}':
            if stack:
                stack.pop()
            if not stack:
                if 'ext{' in text[start_index:i]:
                    return extract_text(text[start_index:i])
                return text[start_index + 1:i].strip()
    return default_value if default_value else text[start_index + 1:].strip()

def extract_belief_state(text, old_scratch_pad, valid_actions=None):
    scratch_pad = extract_boxed(text.split("</think>")[-1], default_value="").strip()
    if scratch_pad != "":
        scratch_pad = re.sub(r'[^' + valid_actions + ']', '', scratch_pad)
        if scratch_pad != "":
            old_scratch_pad = scratch_pad
    return old_scratch_pad