import re


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'([0123456789\-\+\=\*\<\>\;\:\|\n])', r' ', text)
    text = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', text)
    text = text.replace('&', ' and ')
    text = text.replace('@', ' at ')
    return text
