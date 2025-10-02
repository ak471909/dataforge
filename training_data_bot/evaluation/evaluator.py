"""
Quality evaluator for training data.

This module provides comprehensive quality assessment for training examples
and datasets using multiple metrics.
"""

from typing import Dict, List, Optional, Any
from uuid import uuid4

from training_data_bot.core import (
    TrainingExample,
    Dataset,
    QualityReport,
    QualityMetric,
    get_logger,
    LogContext,
)


class QualityEvaluator:
    """
    Quality evaluator for training data.
    
    Assesses the quality of training examples and datasets using
    multiple metrics including relevance, coherence, diversity, etc.
    """
    
    def __init__(
        self,
        quality_threshold: float = 0.7,
        **kwargs
    ):
        """
        Initialize the quality evaluator.
        
        Args:
            quality_threshold: Minimum quality score to pass (0.0-1.0)
            **kwargs: Additional configuration parameters
        """
        self.logger = get_logger("evaluation.QualityEvaluator")
        self.quality_threshold = quality_threshold
        self.config = kwargs
        
        # Metric weights for overall score calculation
        self.metric_weights = {
            QualityMetric.RELEVANCE: 0.25,
            QualityMetric.COHERENCE: 0.20,
            QualityMetric.COMPLETENESS: 0.20,
            QualityMetric.DIVERSITY: 0.15,
            QualityMetric.ACCURACY: 0.10,
            QualityMetric.TOXICITY: 0.05,  # Negative metric (lower is better)
            QualityMetric.BIAS: 0.05,      # Negative metric (lower is better)
        }
        
        self.logger.info(
            "QualityEvaluator initialized",
            threshold=quality_threshold
        )
    
    def evaluate_example(
        self,
        example: TrainingExample,
        detailed: bool = True
    ) -> QualityReport:
        """
        Evaluate a single training example.
        
        Args:
            example: Training example to evaluate
            detailed: Whether to include detailed analysis
            
        Returns:
            QualityReport with evaluation results
        """
        with LogContext("evaluate_example", example_id=str(example.id)):
            self.logger.debug(f"Evaluating example {example.id}")
            
            # Calculate individual metrics
            metric_scores = {}
            
            # Relevance: How well output relates to input
            metric_scores[QualityMetric.RELEVANCE] = self._assess_relevance(
                example.input_text,
                example.output_text
            )
            
            # Coherence: Internal consistency and logical flow
            metric_scores[QualityMetric.COHERENCE] = self._assess_coherence(
                example.output_text
            )
            
            # Completeness: Output is substantial and complete
            metric_scores[QualityMetric.COMPLETENESS] = self._assess_completeness(
                example.output_text
            )
            
            # Accuracy: Based on existing quality scores
            metric_scores[QualityMetric.ACCURACY] = self._assess_accuracy(
                example.quality_scores
            )
            
            # Toxicity: Check for harmful content (lower is better)
            metric_scores[QualityMetric.TOXICITY] = self._assess_toxicity(
                example.output_text
            )
            
            # Bias: Check for biased language (lower is better)
            metric_scores[QualityMetric.BIAS] = self._assess_bias(
                example.output_text
            )
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(metric_scores)
            
            # Determine if passed
            passed = overall_score >= self.quality_threshold
            
            # Generate issues and recommendations
            issues = []
            warnings = []
            recommendations = []
            
            if detailed:
                issues, warnings, recommendations = self._generate_feedback(
                    metric_scores,
                    example
                )
            
            # Create quality report
            report = QualityReport(
                id=uuid4(),
                target_id=example.id,
                target_type="example",
                overall_score=overall_score,
                passed=passed,
                metric_scores=metric_scores,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations
            )
            
            self.logger.debug(
                f"Example evaluation complete",
                overall_score=overall_score,
                passed=passed
            )
            
            return report
    
    def evaluate_dataset(
        self,
        dataset: Dataset,
        detailed_report: bool = True
    ) -> QualityReport:
        """
        Evaluate an entire dataset.
        
        Args:
            dataset: Dataset to evaluate
            detailed_report: Whether to include detailed analysis
            
        Returns:
            QualityReport with dataset evaluation results
        """
        with LogContext("evaluate_dataset", dataset_id=str(dataset.id)):
            self.logger.info(
                f"Evaluating dataset with {len(dataset.examples)} examples"
            )
            
            if not dataset.examples:
                return QualityReport(
                    id=uuid4(),
                    target_id=dataset.id,
                    target_type="dataset",
                    overall_score=0.0,
                    passed=False,
                    metric_scores={},
                    issues=["Dataset is empty"],
                    warnings=[],
                    recommendations=["Add training examples to the dataset"]
                )
            
            # Evaluate individual examples
            example_reports = []
            for example in dataset.examples:
                report = self.evaluate_example(example, detailed=False)
                example_reports.append(report)
            
            # Aggregate metrics
            metric_scores = {}
            for metric in QualityMetric:
                scores = [
                    r.metric_scores.get(metric, 0.0)
                    for r in example_reports
                    if metric in r.metric_scores
                ]
                if scores:
                    metric_scores[metric] = sum(scores) / len(scores)
            
            # Dataset-specific metrics
            metric_scores[QualityMetric.DIVERSITY] = self._assess_dataset_diversity(
                dataset
            )
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(metric_scores)
            
            # Determine if passed
            passed = overall_score >= self.quality_threshold
            
            # Generate feedback
            issues = []
            warnings = []
            recommendations = []
            
            if detailed_report:
                issues, warnings, recommendations = self._generate_dataset_feedback(
                    metric_scores,
                    dataset,
                    example_reports
                )
            
            # Create quality report
            report = QualityReport(
                id=uuid4(),
                target_id=dataset.id,
                target_type="dataset",
                overall_score=overall_score,
                passed=passed,
                metric_scores=metric_scores,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations
            )
            
            self.logger.info(
                f"Dataset evaluation complete",
                overall_score=overall_score,
                passed=passed,
                examples_evaluated=len(example_reports)
            )
            
            return report
    
    def _assess_relevance(self, input_text: str, output_text: str) -> float:
        """Assess how relevant output is to input."""
        # Simple word overlap heuristic
        input_words = set(input_text.lower().split())
        output_words = set(output_text.lower().split())
        
        if not input_words:
            return 0.0
        
        overlap = len(input_words & output_words)
        relevance = min(overlap / len(input_words), 1.0)
        
        return relevance
    
    def _assess_coherence(self, text: str) -> float:
        """Assess internal coherence of text."""
        # Simple heuristic based on sentence structure
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Check for reasonable sentence lengths
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Ideal sentence length: 10-25 words
        if 10 <= avg_sentence_length <= 25:
            coherence = 1.0
        elif 5 <= avg_sentence_length <= 35:
            coherence = 0.8
        else:
            coherence = 0.6
        
        return coherence
    
    def _assess_completeness(self, text: str) -> float:
        """Assess if text is complete and substantial."""
        word_count = len(text.split())
        
        if word_count < 10:
            return 0.3
        elif word_count < 20:
            return 0.6
        elif word_count < 50:
            return 0.8
        else:
            return 1.0
    
    def _assess_accuracy(self, quality_scores: Dict[str, float]) -> float:
        """Assess accuracy based on existing quality scores."""
        if not quality_scores:
            return 0.5  # Neutral if no scores
        
        # Average existing quality scores
        return sum(quality_scores.values()) / len(quality_scores)
    
    def _assess_toxicity(self, text: str) -> float:
        """Assess toxicity (simple keyword-based check)."""
        # Simple heuristic - in production, use proper toxicity detection
        toxic_keywords = [
            'hate', 'kill', 'stupid', 'idiot', 'damn', 'hell',
            'offensive', 'attack', 'violent'
        ]
        
        text_lower = text.lower()
        toxic_count = sum(1 for word in toxic_keywords if word in text_lower)
        
        # Lower toxicity score is better (inverted)
        if toxic_count == 0:
            return 0.0  # No toxicity (best)
        elif toxic_count == 1:
            return 0.3
        elif toxic_count == 2:
            return 0.6
        else:
            return 1.0  # High toxicity (worst)
    
    def _assess_bias(self, text: str) -> float:
        """Assess potential bias (simple keyword-based check)."""
        # Simple heuristic - in production, use proper bias detection
        biased_patterns = [
            'always', 'never', 'all', 'none', 'everyone', 'no one',
            'must', 'obviously', 'clearly'
        ]
        
        text_lower = text.lower()
        bias_count = sum(1 for pattern in biased_patterns if pattern in text_lower)
        
        # Lower bias score is better (inverted)
        if bias_count == 0:
            return 0.0  # No bias (best)
        elif bias_count <= 2:
            return 0.3
        elif bias_count <= 4:
            return 0.6
        else:
            return 0.9  # High bias (worst)
    
    def _assess_dataset_diversity(self, dataset: Dataset) -> float:
        """Assess diversity of examples in dataset."""
        if len(dataset.examples) < 2:
            return 0.5
        
        # Check diversity based on unique words
        all_words = set()
        example_word_sets = []
        
        for example in dataset.examples:
            words = set(example.output_text.lower().split())
            example_word_sets.append(words)
            all_words.update(words)
        
        # Calculate average uniqueness
        unique_ratios = []
        for words in example_word_sets:
            if all_words:
                ratio = len(words) / len(all_words)
                unique_ratios.append(ratio)
        
        diversity = sum(unique_ratios) / len(unique_ratios) if unique_ratios else 0.5
        return diversity
    
    def _calculate_overall_score(self, metric_scores: Dict[QualityMetric, float]) -> float:
        """Calculate weighted overall quality score."""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, weight in self.metric_weights.items():
            if metric in metric_scores:
                score = metric_scores[metric]
                
                # Invert negative metrics (toxicity, bias)
                if metric in [QualityMetric.TOXICITY, QualityMetric.BIAS]:
                    score = 1.0 - score
                
                weighted_sum += score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _generate_feedback(
        self,
        metric_scores: Dict[QualityMetric, float],
        example: TrainingExample
    ) -> tuple:
        """Generate issues, warnings, and recommendations."""
        issues = []
        warnings = []
        recommendations = []
        
        # Check each metric
        for metric, score in metric_scores.items():
            if metric == QualityMetric.RELEVANCE and score < 0.5:
                issues.append(f"Low relevance score ({score:.2f})")
                recommendations.append("Ensure output directly addresses the input")
            
            if metric == QualityMetric.COHERENCE and score < 0.6:
                warnings.append(f"Coherence could be improved ({score:.2f})")
                recommendations.append("Review text for logical flow and structure")
            
            if metric == QualityMetric.COMPLETENESS and score < 0.5:
                issues.append(f"Output may be incomplete ({score:.2f})")
                recommendations.append("Provide more substantial and complete responses")
            
            if metric == QualityMetric.TOXICITY and score > 0.5:
                issues.append(f"Potential toxic content detected ({score:.2f})")
                recommendations.append("Review and remove any harmful language")
            
            if metric == QualityMetric.BIAS and score > 0.5:
                warnings.append(f"Potential biased language detected ({score:.2f})")
                recommendations.append("Use more balanced and objective language")
        
        return issues, warnings, recommendations
    
    def _generate_dataset_feedback(
        self,
        metric_scores: Dict[QualityMetric, float],
        dataset: Dataset,
        example_reports: List[QualityReport]
    ) -> tuple:
        """Generate dataset-level feedback."""
        issues = []
        warnings = []
        recommendations = []
        
        # Check dataset size
        if len(dataset.examples) < 10:
            warnings.append(f"Small dataset size ({len(dataset.examples)} examples)")
            recommendations.append("Consider generating more training examples")
        
        # Check failure rate
        failed_count = sum(1 for r in example_reports if not r.passed)
        failure_rate = failed_count / len(example_reports) if example_reports else 0
        
        if failure_rate > 0.3:
            issues.append(f"High failure rate ({failure_rate:.1%})")
            recommendations.append("Review and improve example quality")
        elif failure_rate > 0.1:
            warnings.append(f"Moderate failure rate ({failure_rate:.1%})")
        
        # Check diversity
        if QualityMetric.DIVERSITY in metric_scores:
            diversity = metric_scores[QualityMetric.DIVERSITY]
            if diversity < 0.3:
                issues.append(f"Low diversity score ({diversity:.2f})")
                recommendations.append("Generate more varied training examples")
        
        return issues, warnings, recommendations