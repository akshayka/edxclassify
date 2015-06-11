CUSTOM_TOKEN_PATTERNS = [
    # Words are combinations of two or more alphanumeric characters,
    # or a single punctuation character.
    r"(?u)\b\w\w+\b",              # 0 Default regex from documentation
    r"(?u)(\b\w\w+\b|[.,;!?])",    # 1 Custom regex that includes punctuation
    r"(?u)(\b\w\w+\b|[.!?])",      # 2 Custom regex without commas/semicolons
    r"(?u)(\b\w\w+\b|[.,!?])",     # 3 Custom regex with commas
    r"(?u)(\b\w\w+\b|[.;!?])"      # 4 Custom regex with semicolons

    # Words are combinations of one or more alphanumeric characters,
    # or a single punctuation character.
    r"(?u)(\b\w+\b|[.,;!?])",      # 5 Custom regex that includes punctuation
    r"(?u)(\b\w+\b|[.!?])",        # 6 Custom regex without commas/semicolons
    r"(?u)(\b\w+\b|[.,!?])",       # 7 Custom regex with commas
    r"(?u)(\b\w+\b|[.;!?])"        # 8 Custom regex with semicolons
]
