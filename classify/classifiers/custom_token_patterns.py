CUSTOM_TOKEN_PATTERNS = [
    r"(?u)\b\w\w+\b",              # Default regex from documentation
    r"(?u)(\b\w\w+\b|[.,;!?])",    # Custom regex that includes punctuation
    r"(?u)(\b\w\w+\b|[.!?])",      # Custom regex without commas/semicolons
    r"(?u)(\b\w\w+\b|[.,!?])",     # Custom regex with commas
    r"(?u)(\b\w\w+\b|[.;!?])"      # Custom regex with semicolons
]
