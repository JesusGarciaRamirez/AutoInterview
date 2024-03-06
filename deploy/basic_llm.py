import modal
from modal import Image, gpu

GPU_CONFIG = gpu.A100(memory=80, count=1)
MODEL_REPOS = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
MODEL_FILENAME = "mistral-7b-instruct-v0.1.Q6_K.gguf"
MODEL_DIR = "/model"


def download_model():
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo_id=MODEL_REPOS, filename=MODEL_FILENAME, local_dir=MODEL_DIR)


image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .pip_install(
        "huggingface_hub==0.19.4",
        "hf-transfer==0.1.4",
        "torch==2.1.2",
        "langchain~=0.0.138",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands(
        'CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python',
    )
    .run_function(download_model)
)

stub = modal.Stub("mistral_langchain_test_new", image=image)


@stub.cls(gpu=GPU_CONFIG)
class Llm:
    @modal.enter()
    def load_model(self):
        from langchain_community.llms import LlamaCpp

        n_gpu_layers = -1
        n_batch = 512

        self.llm = LlamaCpp(
            model_path=MODEL_DIR + "/" + MODEL_FILENAME,
            temperature=0,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            verbose=False,
        )

    @modal.method()
    def predict(self, question):
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate

        template = """Question: {question}
        Answer: Let's work this out in a step by step way to be sure we have the right answer.
        """
        prompt = PromptTemplate.from_template(template)
        chain = LLMChain(prompt=prompt, llm=self.llm)
        return chain.invoke(question)["text"]


@stub.local_entrypoint()
def main():
    llm = Llm()
    result = llm.predict.remote(
        "What NFL team won the Super Bowl in the year Michael Jackson was born?"
    )
