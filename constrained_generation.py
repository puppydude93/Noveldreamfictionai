import math

# --- Hyperparameters defined in the study ---
# The scaling factor omega is set to 30 based on qualitative inspection to ensure the
# model adapts its behavior without impairing writing quality [5].
OMEGA = 30

def f_omega(x, alpha):
    """
    Transformation function that applies a scaled transformation to delta [2].
    Alpha is calibrated so that |f_omega(x)| is approximately 0.99 * omega when |x| = 1 [5].
    """
    return OMEGA * math.tanh(alpha * x)

def calculate_delta(f_star, f_c):
    """
    Deviation metric calculated in an LSM-inspired fashion to quantify the divergence
    between the desired frequency (f_star) and the current frequency (f_c) [2, 6].
    """
    return (f_star - f_c) / (f_star + f_c)

def logit_constrained_generation(LM, tokens, target_categories, f_stars, Nmax, alpha):
    """
    Algorithm 1: Constrained generation pseudocode [3].

    LM: The Language Model
    tokens: The initial sequence of generated tokens (t1...tn)
    target_categories: Sets of token indices representing predefined categories (Ti)
    f_stars: The desired target frequencies for each category
    Nmax: The maximum number of generated tokens
    """
    n = len(tokens)

    while n < Nmax:
        # 1. Output the current logits from the model [3]
        logits = LM.get_logits(tokens)

        # 2. Iterate through each predefined target category [3]
        for idx, T in enumerate(target_categories):

            # Calculate the current frequency (f_c) of the category in the text [3]
            category_count = sum(1 for t in tokens if t in T)
            f_c = category_count / n

            # Calculate the deviation between current and target frequency [3]
            delta = calculate_delta(f_stars[idx], f_c)

            # 3. Alter the logits based on the divergence [2, 3]
            # This increases the likelihood of generating tokens for underused categories
            # and decreases it for overused ones [7].
            logits += f_omega(delta, alpha)

        # 4. Sample the next token using the altered logits [3]
        next_token = LM.sample(logits)
        tokens.append(next_token)
        n += 1

    return tokens
