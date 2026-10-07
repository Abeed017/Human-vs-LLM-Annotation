# Annotation Guidelines

## Purpose
This document defines tasks, label definitions, and annotation rules for the NLP pipeline.

## Labels

### Sentiment
- Positive: expresses a positive opinion, praise, or optimism.
- Negative: expresses criticism, sadness, anger, or disapproval.
- Neutral: factual statements, questions, or balanced content without strong emotion.

### Emotion
- Happiness: joy, excitement, gratitude, amusement.
- Sadness: sorrow, disappointment, grief, loneliness.
- Anger: irritation, frustration, rage, annoyance.
- Fear: anxiety, worry, panic, concern.
- Love: affection, fondness, loyalty, longing.

### Topic
- Current Affairs: politics, policy, governance, public events.
- Economics: finance, markets, business, trade.
- Geopolitics: international relations, diplomacy, conflict.
- Culture: arts, entertainment, lifestyle, local traditions.
- Sports: competitive games, athletes, tournaments.

## Annotation rules

1. Annotate only the main sentiment or emotion expressed in the text.
2. If the text is ambiguous, choose the label that best matches the overall tone.
3. Remove URLs and bot artifacts before annotation.
4. In Indonesian tweets, try to recognize slang, sarcasm, and regional terms.

## Example annotations

- "Saya sangat senang hari ini" → Happiness
- "Ekonomi negara sedang sulit" → Negative / Economics
- "Ini berita dari dunia politik" → Current Affairs
