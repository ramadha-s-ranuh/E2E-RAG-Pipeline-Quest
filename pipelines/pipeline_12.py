import asyncio
from gllm_pipeline.steps import step
from gllm_pipeline.pipeline import Pipeline
from gllm_pipeline.steps import step


def create_adaptive_pipeline(user_preferences, debug_mode=False):
    """Create a pipeline with exclusions based on user preferences and debug mode."""

    pipeline = Pipeline(
        [
            step(
                DocumentExtractor(),
                {"document": "input_document"},
                "extracted_text",
                name="extract",
            ),
            step(
                SentimentAnalyzer(),
                {"text": "extracted_text"},
                "sentiment_score",
                name="sentiment",
            ),
            step(
                TopicDetector(),
                {"text": "extracted_text"},
                "detected_topics",
                name="topics",
            ),
            step(
                EntityExtractor(),
                {"text": "extracted_text"},
                "named_entities",
                name="entities",
            ),
            step(
                LanguageDetector(),
                {"text": "extracted_text"},
                "language_info",
                name="language",
            ),
            step(
                ReportGenerator(),
                {
                    "sentiment": "sentiment_score",
                    "topics": "detected_topics",
                    "entities": "named_entities",
                    "language": "language_info",
                },
                "analysis_report",
                name="report",
            ),
        ]
    )

    # Apply exclusions based on user preferences
    exclusions = []

    if not user_preferences.get("include_sentiment", True):
        exclusions.append("sentiment")

    if not user_preferences.get("include_entities", True):
        exclusions.append("entities")

    if debug_mode:
        # In debug mode, skip expensive operations
        exclusions.extend(["topics", "language"])

    if exclusions:
        pipeline.exclusions.exclude(*exclusions)

    return pipeline


# Usage
if "__main__" == __name__:
    user_prefs = {"include_sentiment": False, "include_entities": True}
    pipeline = create_adaptive_pipeline(user_prefs, debug_mode=True)
    result = await pipeline.invoke({"input_document": "Sample content"})
