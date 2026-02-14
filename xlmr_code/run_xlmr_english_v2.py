import argparse
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import os

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')
    qwk = cohen_kappa_score(labels, preds, weights='quadratic')
    
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'qwk': qwk
    }

class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)
            
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser(description='Run XLM-R English Experiment V2 (Optimized)')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the JSON dataset file')
    parser.add_argument('--output_dir', type=str, default='./results_xlmr_en_v2', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per device')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs (Increased)')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate (Optimized)')
    parser.add_argument('--max_length', type=int, default=256, help='Max sequence length (Increased)')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience (Increased)')
    
    args = parser.parse_args()
    
    # Load Data
    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Ensure labels are 0-4
    if 'confidentiality_degree' in df.columns:
        df['label'] = df['confidentiality_degree'] - 1
    elif 'label' in df.columns:
        if df['label'].min() >= 1:
             df['label'] = df['label'] - 1
    else:
        raise ValueError("Input JSON must contain 'confidentiality_degree' or 'label' field.")

    # Check for English text column
    if 'english_email' in df.columns:
        text_col = 'english_email'
    elif 'text_en' in df.columns:
        text_col = 'text_en'
    else:
        raise ValueError("Input JSON must contain 'english_email' or 'text_en' field.")

    print(f"Using text column: {text_col}")
    
    # Compute Class Weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(df['label']),
        y=df['label']
    )
    print(f"Computed Class Weights: {class_weights}")
    
    # Tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    # 5-Fold Stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold_idx, (train_index, test_index) in enumerate(skf.split(df, df['label'])):
        print(f"\n=== Running Fold {fold_idx + 1}/5 ===")
        
        train_texts = df.iloc[train_index][text_col].tolist()
        train_labels = df.iloc[train_index]['label'].tolist()
        
        test_texts = df.iloc[test_index][text_col].tolist()
        test_labels = df.iloc[test_index]['label'].tolist()
        
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=args.max_length)
        test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=args.max_length)
        
        train_dataset = EmailDataset(train_encodings, train_labels)
        test_dataset = EmailDataset(test_encodings, test_labels)
        
        model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=5)
        
        training_args = TrainingArguments(
            output_dir=f'{args.output_dir}/fold_{fold_idx}',
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            warmup_steps=100,
            weight_decay=0.01,
            learning_rate=args.lr,
            logging_dir=f'{args.output_dir}/logs',
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            seed=42,
            data_seed=42,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=2,
            dataloader_pin_memory=False # Fix for MPS warning
        )
        
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
        )
        
        trainer.train()
        
        # Evaluate
        eval_result = trainer.evaluate()
        print(f"Fold {fold_idx + 1} Results: {eval_result}")
        fold_results.append(eval_result)
        
        # Save fold results
        with open(f'{args.output_dir}/results_fold_{fold_idx}.json', 'w') as f:
            json.dump(eval_result, f, indent=4)

    # Aggregate Results
    avg_acc = np.mean([res['eval_accuracy'] for res in fold_results])
    avg_f1 = np.mean([res['eval_f1_macro'] for res in fold_results])
    avg_qwk = np.mean([res['eval_qwk'] for res in fold_results])
    
    print("\n=== Final Optimized 5-Fold CV Results ===")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average F1-Macro: {avg_f1:.4f}")
    print(f"Average QWK: {avg_qwk:.4f}")
    
    final_results = {
        'avg_accuracy': avg_acc,
        'avg_f1_macro': avg_f1,
        'avg_qwk': avg_qwk,
        'fold_results': fold_results
    }
    
    with open(f'{args.output_dir}/final_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
        
    print(f"Results saved to {args.output_dir}/final_results.json")

if __name__ == "__main__":
    main()
