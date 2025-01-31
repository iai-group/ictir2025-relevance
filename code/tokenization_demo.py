import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from transformers import BertTokenizer  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print(tokenizer("What is your name?", "My name is Ole."))
