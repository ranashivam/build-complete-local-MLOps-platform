from kfp import dsl
from kfp import compiler


@dsl.component
def add_numbers(a: int, b: int) -> int:
    return a + b


@dsl.pipeline(
    name="simple-pipeline"
)
def my_pipeline():

    add_numbers(
        a=10,
        b=20
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=my_pipeline,
        package_path="simple_pipeline.yaml"
    )

    print("Pipeline compiled successfully!")
    print("Generated file: simple_pipeline.yaml")