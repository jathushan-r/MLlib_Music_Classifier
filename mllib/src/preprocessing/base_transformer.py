"""
Base transformer class for all text preprocessing transformers.
"""
from abc import abstractmethod

from pyspark.sql import DataFrame
from pyspark.ml import Transformer
from pyspark.ml.util import (
    DefaultParamsReader,
    DefaultParamsWriter,
    MLReadable,
    MLReader,
    MLWritable,
    MLWriter,
)


class BaseTransformer(Transformer, MLReadable, MLWritable):
    """
    Base class for all lyrics text transformers in the preprocessing pipeline.
    
    This abstract class implements MLReadable and MLWritable interfaces to support
    serialization and deserialization of transformers in the Spark ML pipeline.
    """
    
    @abstractmethod
    def _transform(self, dataframe: DataFrame) -> DataFrame:
        """
        Transformation logic to be implemented by subclasses.
        
        Args:
            dataframe: Input DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        pass

    def transform(self, dataframe: DataFrame) -> DataFrame:
        """
        Transform the input DataFrame.
        
        Args:
            dataframe: Input DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        return self._transform(dataframe)

    def write(self) -> MLWriter:
        """
        Write this transformer to disk.
        
        Returns:
            MLWriter instance for writing
        """
        return DefaultParamsWriter(self)

    @classmethod
    def read(cls) -> MLReader:
        """
        Read this transformer from disk.
        
        Returns:
            MLReader instance for reading
        """
        return DefaultParamsReader(cls)
    