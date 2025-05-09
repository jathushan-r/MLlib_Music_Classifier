"""
Text cleaning transformer for lyrics data.
"""
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import regexp_replace, trim, col

from src.utils.constants import Column
from src.preprocessing.base_transformer import BaseTransformer


class Cleanser(BaseTransformer):
    """
    Transformer for cleaning raw lyrics text.
    
    This transformer:
    1. Removes punctuation and special characters
    2. Trims leading/trailing whitespace
    3. Replaces multiple spaces with single space
    4. Filters out null values
    """
    
    def _transform(self, dataframe: DataFrame) -> DataFrame:
        """
        Clean the lyrics text in the input DataFrame.
        
        Args:
            dataframe: Input DataFrame with raw lyrics text
            
        Returns:
            DataFrame with cleaned lyrics text
        """
        # Remove punctuation and special characters
        dataframe = dataframe.withColumn(
            Column.CLEAN.value, 
            regexp_replace(trim(col(Column.VALUE.value)), r"[^\w\s]", "")
        )
        
        # Replace multiple spaces with single space
        dataframe = dataframe.withColumn(
            Column.CLEAN.value, 
            regexp_replace(col(Column.CLEAN.value), r"\s{2,}", " ")
        )
        
        # Drop original raw text column
        dataframe = dataframe.drop(Column.VALUE.value)
        
        # Filter out null values
        dataframe = dataframe.filter(col(Column.CLEAN.value).isNotNull())
        
        return dataframe
    