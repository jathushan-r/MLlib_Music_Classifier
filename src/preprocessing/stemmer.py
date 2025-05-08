"""
Word stemming transformer for lyrics data.
"""
from nltk.stem import SnowballStemmer
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, StringType

from src.utils.constants import Column
from src.preprocessing.base_transformer import BaseTransformer


class Stemmer(BaseTransformer):
    """
    Transformer for stemming tokenized words in lyrics.
    
    This transformer applies the Snowball stemmer to reduce words to their root forms,
    which helps improve classification by normalizing similar words.
    """
    
    def _transform(self, dataframe: DataFrame) -> DataFrame:
        """
        Apply stemming to tokenized words in the input DataFrame.
        
        Args:
            dataframe: Input DataFrame with tokenized lyrics
            
        Returns:
            DataFrame with stemmed words
        """
        # Initialize the Snowball stemmer for English
        stemmer = SnowballStemmer("english")
        
        # Define UDF for stemming
        stem_udf = udf(
            lambda words: [stemmer.stem(word) for word in words], 
            ArrayType(StringType())
        )
        
        # Apply stemming to filtered words
        dataframe = dataframe.withColumn(
            Column.STEMMED_WORDS.value, 
            stem_udf(col(Column.FILTERED_WORDS.value))
        )
        
        # Select only necessary columns
        dataframe = dataframe.select(Column.STEMMED_WORDS.value, Column.LABEL.value)
        
        return dataframe