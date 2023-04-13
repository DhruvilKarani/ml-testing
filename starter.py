import tensorflow as tf
print('TensorFlow version: {}'.format(tf.__version__))
from tfx import v1 as tfx
print('TFX version: {}'.format(tfx.__version__))
from absl import logging
import os
import urllib.request
import tempfile


# We will create two pipelines. One for schema generation and one for training.
SCHEMA_PIPELINE_NAME = "penguin-tfdv-schema"
PIPELINE_NAME = "penguin-tfdv"

# Output directory to store artifacts generated from the pipeline.
SCHEMA_PIPELINE_ROOT = os.path.join('pipelines', SCHEMA_PIPELINE_NAME)
PIPELINE_ROOT = os.path.join('pipelines', PIPELINE_NAME)
# Path to a SQLite DB file to use as an MLMD storage.
SCHEMA_METADATA_PATH = os.path.join('metadata', SCHEMA_PIPELINE_NAME,
                                    'metadata.db')
METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')

# Output directory where created models from the pipeline will be exported.
SERVING_MODEL_DIR = os.path.join('serving_model', PIPELINE_NAME)

logging.set_verbosity(logging.INFO)  # Set default logging level.


DATA_ROOT = tempfile.mkdtemp(prefix='tfx-data')  # Create a temporary directory.
_data_url = 'https://raw.githubusercontent.com/tensorflow/tfx/master/tfx/examples/penguin/data/labelled/penguins_processed.csv'
_data_filepath = os.path.join(DATA_ROOT, "data.csv")
urllib.request.urlretrieve(_data_url, _data_filepath)

print("Data downloaded to {}.".format(_data_filepath))

def _create_schema_pipeline(pipeline_name: str,
                            pipeline_root: str,
                            data_root: str,
                            metadata_path: str) -> tfx.dsl.Pipeline:
    """Creates a pipeline for schema generation."""
    # Brings data into the pipeline.
    example_gen = tfx.components.CsvExampleGen(input_base=data_root)

    # Computes statistics over data for visualization and example validation.
    statistics_gen = tfx.components.StatisticsGen(examples=example_gen.outputs['examples'])

    # Generates schema based on statistics files.
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=False)
    
    # Uses user-provided Python function that implements a schema
    # validator.
    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])
    

    
    # Creates the pipeline.
    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
    ]
    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path),
    )
    

tfx.orchestration.LocalDagRunner().run(
  _create_schema_pipeline(
      pipeline_name=SCHEMA_PIPELINE_NAME,
      pipeline_root=SCHEMA_PIPELINE_ROOT,
      data_root=DATA_ROOT,
      metadata_path=SCHEMA_METADATA_PATH))
