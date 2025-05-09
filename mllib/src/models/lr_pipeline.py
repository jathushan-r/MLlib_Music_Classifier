"""
Logistic regression pipeline for lyrics classification.
"""
from typing import Dict, Any, cast, List

from pyspark.sql.dataframe import DataFrame
from pyspark.ml import Pipeline, PipelineModel, Transformer
from pyspark.ml.feature import StopWordsRemover, Tokenizer, Word2Vec
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import (
    CrossValidator,
    CrossValidatorModel,
    ParamGridBuilder,
)

from src.utils.constants import Column
from src.models.base_pipeline import BasePipeline
from src.preprocessing.cleanser import Cleanser
from src.preprocessing.label_encoder import LabelEncoder
from src.preprocessing.stemmer import Stemmer


class LogisticRegressionPipeline(BasePipeline):
    """
    Logistic regression pipeline for lyrics classification.
    
    This pipeline:
    1. Encodes genre labels
    2. Cleans text
    3. Tokenizes text
    4. Removes stop words
    5. Stems words
    6. Converts words to vectors
    7. Trains a logistic regression model
    """
    
    def train(
        self,
        dataframe: DataFrame,
        print_statistics: bool = False,
    ) -> CrossValidatorModel:
        """
        Train a logistic regression model on the provided data.
        
        Args:
            dataframe: Training data
            print_statistics: Whether to print model statistics
            
        Returns:
            Trained CrossValidatorModel
        """
        # Select only the needed columns
        dataframe = dataframe.select(Column.VALUE.value, Column.GENRE.value)

        # Define pipeline stages
        label_encoder = LabelEncoder()
        cleanser = Cleanser()
        tokenizer = Tokenizer(
            inputCol=Column.CLEAN.value,
            outputCol=Column.WORDS.value,
        )
        stop_words_remover = StopWordsRemover(
            inputCol=Column.WORDS.value,
            outputCol=Column.FILTERED_WORDS.value,
        )
        stemmer = Stemmer()
        word_to_vec = Word2Vec(
            inputCol=Column.STEMMED_WORDS.value,
            outputCol=Column.FEATURES.value,
            minCount=0,
            seed=42,
        )
        lr = LogisticRegression(
            featuresCol=Column.FEATURES.value,
            labelCol=Column.LABEL.value,
            predictionCol=Column.PREDICTION.value,
            probabilityCol=Column.PROBABILITY.value,
        )

        # Create pipeline
        pipeline = Pipeline(
            stages=[
                label_encoder,
                cleanser,
                tokenizer,
                stop_words_remover,
                stemmer,
                word_to_vec,
                lr,
            ]
        )

        # Define parameter grid for hyperparameter tuning
        param_grid_builder = ParamGridBuilder()
        param_grid_builder.addGrid(word_to_vec.vectorSize, [500])
        param_grid_builder.addGrid(lr.regParam, [0.01])
        param_grid_builder.addGrid(lr.maxIter, [100])
        param_grid = param_grid_builder.build()

        # Define evaluator
        evaluator = MulticlassClassificationEvaluator(
            predictionCol=Column.PREDICTION.value,
            labelCol=Column.LABEL.value,
            metricName="f1",
        )

        # Create cross validator
        cross_validator = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=5,
            seed=42,
        )

        # Fit model
        cross_validator_model = cross_validator.fit(dataframe)

        # Print statistics if requested
        if print_statistics:
            print(f"MODEL STATISTICS: {self.get_model_statistics(cross_validator_model)}")

        return cross_validator_model

    def get_model_statistics(self, model: CrossValidatorModel) -> Dict[str, Any]:
        """
        Get statistics for a trained model.
        
        Args:
            model: Trained CrossValidatorModel
            
        Returns:
            Dictionary of model statistics
        """
        # Get basic statistics
        model_statistics = self.get_model_basic_statistics(model)

        # Get details from the best model
        best_model: PipelineModel = cast(PipelineModel, model.bestModel)
        stages: List[Transformer] = best_model.stages

        # Add logistic regression parameters
        model_statistics["RegParam"] = cast(LogisticRegression, stages[-1]).getRegParam()
        model_statistics["MaxIter"] = cast(LogisticRegression, stages[-1]).getMaxIter()
        
        # Add Word2Vec parameters
        model_statistics["VectorSize"] = cast(Word2Vec, stages[-2]).getVectorSize()

        return model_statistics
    