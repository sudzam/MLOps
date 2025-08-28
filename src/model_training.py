import os
os.environ.pop("MLFLOW_EXPERIMENT_ID", None)

import mlflow
import mlflow.pytorch
from config import MODEL_NAME, MODEL_DIR, MAX_LENGTH, BATCH_SIZE, NUM_EPOCHS
import os
from time import sleep

# create maseter class for training
class GPT2Trainer:
    def __init__(self):
        print ("Initializing GPT2Trainer...")
        import torch
        print ("Importing transformers...")
        from transformers import GPT2LMHeadModel
        from transformers import GPT2Tokenizer
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        sleep(10)
        self.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        
    # Create dataset for training
    def create_dataset(self, text_file):
        dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=text_file,
            block_size=MAX_LENGTH
        )
        return dataset
    
    # Fine-tune GPT-2 model
    def train_model(self, train_data_path, output_dir):
        # Prepare training data
        with open(train_data_path.replace('.csv', '.txt'), 'w') as f:
            import pandas as pd
            df = pd.read_csv(train_data_path)
            for text in df['formatted_text']:
                f.write(text + '\n')
        
        train_dataset = self.create_dataset(train_data_path.replace('.csv', '.txt'))
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Training arguments optimized for my machine..
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            #use_cpu=True,  # Force CPU usage
            dataloader_num_workers=0,  # Prevent multiprocessing issues on CPU
        )
        
        trainer = Trainer(
            model=self.model.to(self.device),
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )
        
        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer

# Train with MLflow
def train_and_log_model(train_data_path, model_version="1.0"):
    print ("Starting MLflow run inside train_and_log_model")
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("max_length", MAX_LENGTH)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("num_epochs", NUM_EPOCHS)
        mlflow.log_param("model_version", model_version)
        
        # Train model
        print("Training GPT2Trainer")
        trainer_obj = GPT2Trainer()
        model_output_dir = os.path.join(MODEL_DIR, f"gpt2_sentiment_v{model_version}")
        
        trainer = trainer_obj.train_model(train_data_path, model_output_dir)
        
        print("Training complete, logging metrics and model to MLflow")
        # Log metrics (you can add validation metrics here)
        final_loss = trainer.state.log_history[-1].get('train_loss', 0)
        mlflow.log_metric("final_train_loss", final_loss)
        
        # Log model
        mlflow.pytorch.log_model(trainer_obj.model, "model")
        
        print(f"Model trained and saved to {model_output_dir}")
        return model_output_dir


"""
    GPT2LMHeadModel,
        GPT2Tokenizer,   
        TextDataset, 
        DataCollatorForLanguageModeling,
        Trainer, 
        TrainingArguments)
    """