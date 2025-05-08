"""
Base pipeline for lyrics classification models.
"""
from abc import abstractmethod
from typing import Optional, Dict, Tuple, cast, Any

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import lit
from pyspark.ml import PipelineModel
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from src.utils.constants import Column
from src.utils.genre_mapper import LABEL_TO_GENRE_MAP


class BasePipeline:
    """
    Base class for all lyrics classification pipelines.
    
    This abstract class provides common functionality for:
    1. Spark session management
    2. Data loading and splitting
    3. Model evaluation
    4. Prediction
    """
    
    def __init__(self, app_name: str = "LyricsClassifier") -> None:
        """
        Initialize the pipeline with a Spark session.
        
        Args:
            app_name: Name for the Spark application
        """
        print("STARTING SPARK SESSION")
        self.spark = SparkSession.builder\
            .appName(app_name)\
            .config("spark.driver.memory", "3G")\
            .config("spark.executor.memory", "3G")\
            .config("spark.executor.cores", "3")\
            .config("spark.python.worker.memory", "3G") \
            .config("spark.driver.port", "4040")\
            .getOrCreate()

        self.spark.sparkContext.setLogLevel("ERROR")

    def stop(self) -> None:
        """Stop the Spark session."""
        print("STOPPING SPARK SESSION")
        self.spark.stop()

    def read_csv(self, path: str) -> DataFrame:
        """
        Read CSV data into a Spark DataFrame.
        
        Args:
            path: Path to the CSV file
            
        Returns:
            Spark DataFrame containing the data
        """
        return self.spark.read.csv(path, header=True, inferSchema=True)

    def train_and_test(
        self,
        dataset_path: str,
        train_ratio: float = 0.8,
        store_model_on: Optional[str] = None,
        print_statistics: bool = False,
    ) -> CrossValidatorModel:
        """
        Train and test a model on the provided dataset.
        
        Args:
            dataset_path: Path to the dataset CSV file
            train_ratio: Ratio of data to use for training (default: 0.8)
            store_model_on: Directory to save the trained model (default: None)
            print_statistics: Whether to print model statistics (default: False)
            
        Returns:
            Trained CrossValidatorModel
        """
        # Load and split the data
        data = self.read_csv(dataset_path)
        train_df, test_df = data.randomSplit([train_ratio, (1 - train_ratio)], seed=42)

        # Train the model
        model: CrossValidatorModel = self.train(train_df, print_statistics)
        
        # Test the model
        test_accuracy: float = self.test(test_df, model)

        # Print statistics if requested
        if print_statistics:
            print(f"CROSS VALIDATOR MODEL AVERAGE METRICS: {model.avgMetrics}")
            print(f"TEST ACCURACY: {test_accuracy}")

        # Save the model if requested
        if store_model_on:
            model.write().overwrite().save(store_model_on)

        return model

    @abstractmethod
    def train(
        self, 
        dataframe: DataFrame, 
        print_statistics: bool = False
    ) -> CrossValidatorModel:
        """
        Train a model on the provided data.
        
        Args:
            dataframe: Training data
            print_statistics: Whether to print model statistics
            
        Returns:
            Trained CrossValidatorModel
        """
        pass

    def test(
        self,
        dataframe: DataFrame,
        model: Optional[CrossValidatorModel] = None,
        saved_model_dir_path: Optional[str] = None,
    ) -> float:
        """
        Test a model on the provided data.
        
        Args:
            dataframe: Test data
            model: Trained model (default: None, requires saved_model_dir_path)
            saved_model_dir_path: Path to saved model (default: None, requires model)
            
        Returns:
            Accuracy of the model on the test data
        """
        # Load the model if not provided
        if not model:
            model = CrossValidatorModel.load(saved_model_dir_path)

        # Get the best model from the cross validator
        best_model: PipelineModel = cast(PipelineModel, model.bestModel)

        # Make predictions
        predictions = best_model.transform(dataframe)

        # Evaluate the predictions
        evaluator = MulticlassClassificationEvaluator(
            predictionCol=Column.PREDICTION.value,
            labelCol=Column.LABEL.value,
            metricName="accuracy",
        )

        accuracy = evaluator.evaluate(predictions)

        return accuracy

    def predict_one(
        self,
        unknown_lyrics: str,
        threshold: float,
        model: Optional[CrossValidatorModel] = None,
        saved_model_dir_path: Optional[str] = None,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Predict the genre of a single lyrics text.
        
        Args:
            unknown_lyrics: Lyrics text to classify
            threshold: Minimum probability threshold for prediction
            model: Trained model (default: None, requires saved_model_dir_path)
            saved_model_dir_path: Path to saved model (default: None, requires model)
            
        Returns:
            Tuple of (predicted genre, probability distribution)
        """
        # Create a DataFrame with the input lyrics
        unknown_lyrics_df = self.spark.createDataFrame(
            [(unknown_lyrics,)], 
            [Column.VALUE.value]
        )
        unknown_lyrics_df = unknown_lyrics_df.withColumn(
            Column.GENRE.value, 
            lit("UNKNOWN")
        )

        # Load the model if not provided
        if not model:
            model = CrossValidatorModel.load(saved_model_dir_path)

        # Get the best model from the cross validator
        best_model: PipelineModel = cast(PipelineModel, model.bestModel)

        # Make prediction
        predictions_df = best_model.transform(unknown_lyrics_df)
        prediction_row = predictions_df.first()

        # Get the predicted label
        prediction = prediction_row[Column.PREDICTION.value]
        prediction = LABEL_TO_GENRE_MAP[prediction]

        # Get the probability distribution if available
        if Column.PROBABILITY.value in predictions_df.columns:
            probabilities = prediction_row[Column.PROBABILITY.value]
            probabilities = dict(zip(LABEL_TO_GENRE_MAP.values(), probabilities))

            # Check if prediction is confident enough
            if probabilities[prediction] < threshold:
                prediction = "unknown"

            return prediction, probabilities

        return prediction, {}

    @staticmethod
    def get_model_basic_statistics(model: CrossValidatorModel) -> Dict[str, Any]:
        """
        Get basic statistics for a trained model.
        
        Args:
            model: Trained CrossValidatorModel
            
        Returns:
            Dictionary of model statistics
        """
        model_statistics = dict()
        model.avgMetrics.sort()
        model_statistics["Best model metrics"] = model.avgMetrics[-1]
        return model_statistics