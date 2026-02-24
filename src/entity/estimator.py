import sys

import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging

class TargetValueMapping:
    def __init__(self):
        self.yes:int = 0
        self.no:int = 1
    def _asdict(self):
        return self.__dict__
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(),mapping_response.keys()))

class MyModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame) -> DataFrame:
        """
        Function accepts preprocessed inputs (with all custom transformations already applied),
        applies scaling using preprocessing_object, and performs prediction on transformed features.
        """
        try:
            logging.info("Starting prediction process.")

            expected_columns = self._get_expected_columns()
            if expected_columns:
                dataframe = self._align_dataframe_to_expected_columns(
                    dataframe=dataframe,
                    expected_columns=expected_columns
                )

            # Step 1: Apply scaling transformations using the pre-trained preprocessing object
            transformed_feature = self.preprocessing_object.transform(dataframe)

            # Step 2: Perform prediction using the trained model
            logging.info("Using the trained model to get predictions")
            predictions = self.trained_model_object.predict(transformed_feature)

            return predictions

        except Exception as e:
            logging.error("Error occurred in predict method", exc_info=True)
            raise MyException(e, sys) from e

    def _get_expected_columns(self):
        """Get the feature columns seen during preprocessor fitting, if available."""
        preprocessor = None

        if hasattr(self.preprocessing_object, "named_steps"):
            preprocessor = self.preprocessing_object.named_steps.get("Preprocessor")

        if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
            return list(preprocessor.feature_names_in_)

        if hasattr(self.preprocessing_object, "feature_names_in_"):
            return list(self.preprocessing_object.feature_names_in_)

        return None

    @staticmethod
    def _align_dataframe_to_expected_columns(
        dataframe: pd.DataFrame, expected_columns: list
    ) -> pd.DataFrame:
        """Align inference dataframe schema with fitted preprocessor schema."""
        aligned_df = dataframe.copy()

        missing_columns = [col for col in expected_columns if col not in aligned_df.columns]
        extra_columns = [col for col in aligned_df.columns if col not in expected_columns]

        if missing_columns:
            logging.info(
                "Adding missing columns for inference with default 0 values: %s",
                missing_columns
            )
            for col in missing_columns:
                aligned_df[col] = 0

        if extra_columns:
            logging.info("Dropping unexpected columns from inference data: %s", extra_columns)
            aligned_df = aligned_df.drop(columns=extra_columns)

        return aligned_df[expected_columns]


    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
