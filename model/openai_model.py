from langchain_openai import ChatOpenAI
from model.base_model import BaseModel


class OpenAIModel(BaseModel):
    """
    Model wrapper for OpenAI's Chat API.
    """

    def __init__(self, model_name, base_url=None, api_key=None, temperature=0.0):
        super().__init__(api_url=base_url, api_key=api_key, temperature=temperature)
        self.llm = ChatOpenAI(
            model=model_name,
            base_url=self.api_url,
            api_key=self.api_key,
            temperature=self.temperature,
        )

    def invoke(self, prompt: str, *args, **kwargs):
        """
        Invoke the OpenAI model with a prompt.

        Args:
            prompt: The input prompt string.
            **kwargs: Additional parameters for the model.

        Returns:
            The model's response as a string.
        """
        try:
            # self.logger.info("Invoking OpenAI model...")
            response = self.llm.invoke(prompt, *args, **kwargs)
            return response
        except Exception as e:
            # self.logger.error(f"Error invoking OpenAI model: {str(e)}", exc_info=True)
            raise RuntimeError("Failed to invoke OpenAI model") from e
