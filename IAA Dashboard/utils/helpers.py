def get_alpha_interpretation(alpha):
    """Get interpretation text for alpha value."""
    if alpha >= 0.8:
        return "Excellent agreement (α ≥ 0.8)"
    elif alpha >= 0.67:
        return "Good agreement (α ≥ 0.67)"
    elif alpha >= 0.4:
        return "Fair agreement (α ≥ 0.4)"
    else:
        return "Poor agreement (α < 0.4)"

def get_agreement_interpretation(alpha):
    """Get text interpretation of alpha value."""
    if alpha >= 0.8:
        return "Excellent agreement"
    elif alpha >= 0.67:
        return "Good agreement"
    elif alpha >= 0.4:
        return "Fair agreement"
    else:
        return "Poor agreement"

def get_comparison(val1, val2):
    """Get comparison text between two values."""
    if val1 > val2:
        return "higher"
    elif val1 < val2:
        return "lower"
    else:
        return "similar"