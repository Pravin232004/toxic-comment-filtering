def compute_severity(predicted_labels: dict) -> int:
    score = 0

    if predicted_labels["toxic"]:
        score += 1
    if predicted_labels["obscene"]:
        score += 1
    if predicted_labels["insult"]:
        score += 1
    if predicted_labels["threat"]:
        score += 2
    if predicted_labels["identity_hate"]:
        score += 2
    if predicted_labels["severe_toxic"]:
        score += 3

    return min(score, 5)

def moderation_action(severity: int) -> str:
    if severity == 0:
        return "Allow"
    elif severity <= 2:
        return "Warn user"
    elif severity <= 4:
        return "Mute user"
    else:
        return "Auto-ban"
