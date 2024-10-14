from deepeval.benchmarks import DROP
from deepeval.benchmarks.tasks import DROPTask
custom_llm = CustomLlama3_8B()

# Define benchmark with specific tasks and shots
benchmark = DROP(
    tasks=[DROPTask.HISTORY_1002],
    n_shots=3
)

# Replace 'mistral_7b' with your own custom model
benchmark.evaluate(model=custom_llm)