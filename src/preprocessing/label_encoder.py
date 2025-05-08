"""
Label encoder transformer for lyrics genre data.
"""
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType

from src.utils.constants import Column
from src.utils.genre_mapper import GENRE_TO_LABEL_MAP
from src.preprocessing.base_transformer import BaseTransformer


class LabelEncoder(BaseTransformer):
    """
    Transformer for encoding genre labels to numeric values.
    
    This transformer converts text genre labels to numeric values
    according to the predefined mapping in genre_mapper.
    """
    
    def _transform(self, dataframe: DataFrame) -> DataFrame:
        """
        Encode genre labels in the input DataFrame.
        
        Args:
            dataframe: Input DataFrame with text genre labels
            
        Returns:
            DataFrame with numeric genre labels
        """
        # Define UDF for encoding genre to label
        genre_to_label_udf = udf(
            lambda genre: GENRE_TO_LABEL_MAP.get(genre.lower(), GENRE_TO_LABEL_MAP["unknown"]), 
            IntegerType()
        )
        
        # Apply genre encoding
        dataframe = dataframe.withColumn(
            Column.LABEL.value, 
            genre_to_label_udf(col(Column.GENRE.value))
        )
        
        # Drop original genre column
        dataframe = dataframe.drop(Column.GENRE.value)
        
        return dataframe