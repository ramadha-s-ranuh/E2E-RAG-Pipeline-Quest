from gllm_pipeline.pipeline import Pipeline
from gllm_pipeline.steps import step, parallel

parallel_pipeline = Pipeline(
    [
        step(DocumentExtractor(), {"document": "input_document"}, "extracted_text"),
        # Run independent operations in parallel
        parallel(
            [
                step(
                    SentimentAnalyzer(), {"text": "extracted_text"}, "sentiment_score"
                ),
                step(TopicDetector(), {"text": "extracted_text"}, "detected_topics"),
                step(EntityExtractor(), {"text": "extracted_text"}, "named_entities"),
                step(LanguageDetector(), {"text": "extracted_text"}, "language_info"),
            ],
            name="content_analysis_parallel",
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
        ),
    ]
)

# Total execution time: ~3.5 seconds (maximum of parallel operations, not sum)
# Performance improvement: 60-70% faster execution!
