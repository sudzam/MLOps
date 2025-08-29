import os
import mlflow
import mlflow.pytorch
import pandas as pd
from config import MODEL_NAME, MODEL_DIR, MAX_LENGTH, BATCH_SIZE, NUM_EPOCHS
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# Remove previous MLFLOW experiment ID if set
os.environ.pop("MLFLOW_EXPERIMENT_ID", None)

class GPT2Trainer:
    def __init__(self):
        """Initializes the trainer with device and model."""
        print("Initializing GPT2Trainer...")
        import torch

        # Select computation device (MPS, CUDA, or CPU)
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")

        # Load tokenizer and model with appropriate settings
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Ensure pad token is set

        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(self.device)

    def create_dataset(self, text_file):
        """Loads and tokenizes the dataset from a text file."""
        try:
            dataset = load_dataset(
                'text',
                data_files=text_file,
                split='train'
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding='max_length',
                max_length=MAX_LENGTH
            )

        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format(type='torch', columns=['input_ids'])  # Ensure torch format for data
        return dataset

    def train_model(self, train_data_path, output_dir):
        """
        Fine-tunes the GPT-2 model with provided training data and saves outputs.
        Args:
            train_data_path (str): Path to training CSV file containing 'formatted_text' column.
            output_dir (str): Directory to save trained model.
        Returns:
            Trainer: The trained Hugging Face Trainer object.
        """
        # Convert CSV to text for easier tokenization if required
        text_path = train_data_path.replace('.csv', '.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            df = pd.read_csv(train_data_path)
            for text in df['formatted_text']:
                f.write(text + '\n')

        train_dataset = self.create_dataset(text_path)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Set training arguments for Hugging Face Trainer
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            dataloader_num_workers=0,
            report_to=["none"],  # Disable other integrations, to avoid logs outside MLflow
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer
        )

        print("Starting training...")
        trainer.train()

        # Save model and tokenizer
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return trainer

def train_and_log_model(train_data_path, model_version="1.0"):
    """
    Runs MLflow experiment for training and logging the GPT-2 model.
    Args:
        train_data_path (str): Path to training CSV file.
        model_version (str): Version identifier for the current model.
    Returns:
        str: Path to the output directory containing the trained model.
    """
    print("Starting MLflow run inside train_and_log_model")
    with mlflow.start_run():
        # Log experiment parameters
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("max_length", MAX_LENGTH)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("num_epochs", NUM_EPOCHS)
        mlflow.log_param("model_version", model_version)

        # Initialize and train model
        print("Training GPT2Trainer...")
        trainer_obj = GPT2Trainer()
        model_output_dir = os.path.join(MODEL_DIR, f"gpt2_sentiment_v{model_version}")

        trainer = trainer_obj.train_model(train_data_path, model_output_dir)

        print("Training complete, logging metrics and model to MLflow")
        # Log metrics
        final_loss = trainer.state.log_history[-1].get("train_loss", 0)
        mlflow.log_metric("final_train_loss", final_loss)

        # Log model using MLflow's PyTorch flavor
        mlflow.pytorch.log_model(trainer_obj.model, "model")

        print(f"Model trained and saved to {model_output_dir}")
        return model_output_dir
