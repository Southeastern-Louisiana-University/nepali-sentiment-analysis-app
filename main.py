import base64

import streamlit as st


import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import unicodedata
from tqdm import tqdm
import os
import logging
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NepaliSentimentDataset(Dataset):
    """Dataset class for Nepali sentiment analysis."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Initialize the dataset.

        Args:
            texts: List of input texts
            labels: List of sentiment labels
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Ensure label is properly converted to integer
        try:
            label = int(label)
            # Validate label is either 0 or 1
            if label not in [0, 1]:
                logger.error(f"Invalid label value {label} at index {idx}")
                raise ValueError(f"Label must be 0 or 1, got {label}")
        except (ValueError, TypeError) as e:
            logger.error(f"Error processing label at index {idx}: {label}")
            raise

        # Preprocess text
        text = self.preprocess_text(text)

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

    @staticmethod
    def preprocess_text(text):
        """Preprocess the input text."""
        if pd.isna(text):
            return ""
        text = unicodedata.normalize('NFKC', str(text))
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        return text.strip()

class NepaliSentimentAnalyzer:
    """Main class for Nepali sentiment analysis."""

    def __init__(self, model_name='bert-base-multilingual-cased', model_dir='nepali_sentiment_model'):
        """
        Initialize the analyzer.

        Args:
            model_name: Name of the pretrained model
            model_dir: Directory to save/load models
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.model_name = model_name
        self.model_dir = model_dir

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Initialize tokenizer and model
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2  # Binary classification: 0 and 1
        ).to(self.device)

    def prepare_data(self, csv_path, test_size=0.2, val_size=0.1):
        """
        Prepare the datasets from CSV file.

        Args:
            csv_path: Path to the CSV file
            test_size: Proportion of test set
            val_size: Proportion of validation set
        """
        # Read CSV file
        logger.info("Loading and preparing data...")
        df = pd.read_csv(csv_path)

        # Convert labels to 0 and 1
        df['Sentiment'] = df['Sentiment'].apply(lambda x: 1 if x == 1 else 0)

        # Check for any invalid labels
        unique_labels = df['Sentiment'].unique()
        logger.info(f"Unique labels in dataset: {unique_labels}")
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError(f"Invalid labels found in dataset: {unique_labels}")

        # Log data statistics
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Label distribution:\n{df['Sentiment'].value_counts()}")

        # First split into train and test
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            df['Sentences'].values,
            df['Sentiment'].values,
            test_size=test_size,
            random_state=42,
            stratify=df['Sentiment']
        )

        # Then split train into train and validation
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts,
            train_labels,
            test_size=val_size/(1-test_size),
            random_state=42,
            stratify=train_labels
        )

        # Create datasets
        train_dataset = NepaliSentimentDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = NepaliSentimentDataset(val_texts, val_labels, self.tokenizer)
        test_dataset = NepaliSentimentDataset(test_texts, test_labels, self.tokenizer)

        logger.info(f"Train size: {len(train_dataset)}")
        logger.info(f"Validation size: {len(val_dataset)}")
        logger.info(f"Test size: {len(test_dataset)}")

        return train_dataset, val_dataset, test_dataset

    def train(self, train_dataset, val_dataset,
              epochs=3,
              batch_size=8,
              learning_rate=2e-5,
              warmup_steps=0,
              weight_decay=0.01,
              save_steps=100):
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for regularization
            save_steps: Save model every n steps
        """
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Prepare optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Training loop
        best_val_loss = float('inf')
        global_step = 0

        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")

            # Training
            self.model.train()
            train_loss = 0
            progress_bar = tqdm(train_loader, desc="Training")

            for batch in progress_bar:
                try:
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)

                    # Forward pass
                    self.model.zero_grad()
                    outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    loss = outputs.loss
                    train_loss += loss.item()

                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    progress_bar.set_postfix({'loss': loss.item()})
                    global_step += 1

                    # Save checkpoint
                    if global_step % save_steps == 0:
                        self.save_model(f"checkpoint-{global_step}")

                except Exception as e:
                    logger.error(f"Error in training batch: {e}")
                    continue

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            val_loss = self.evaluate(val_loader, "Validation")

            logger.info(f"Average training loss: {avg_train_loss:.4f}")
            logger.info(f"Validation loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best_model")

        logger.info("Training completed!")
        return self.load_best_model()

    def evaluate(self, data_loader, desc="Evaluating"):
        """
        Evaluate the model.

        Args:
            data_loader: DataLoader for evaluation
            desc: Description for progress bar
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc=desc)
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)

        # Print classification report
        print("\nClassification Report:")
        report = classification_report(all_labels, all_preds, target_names=['Negative', 'Positive'])
        print(report)

        return avg_loss

    def save_model(self, model_name):
        """Save the model."""
        save_path = os.path.join(self.model_dir, model_name)
        os.makedirs(save_path, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")

    def load_best_model(self):
        """Load the best model."""
        best_model_path = os.path.join(self.model_dir, "best_model")
        if os.path.exists(best_model_path):
            self.model = AutoModelForSequenceClassification.from_pretrained(best_model_path).to(self.device)
            logger.info(f"Loaded best model from {best_model_path}")
        return self.model

    def predict(self, text):
        """
        Predict sentiment for a single text.

        Args:
            text: Input text
        """
        self.model.eval()
        # Preprocess text
        text = NepaliSentimentDataset.preprocess_text(text)

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            prob = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(prob, dim=1).item()

        return {
            'text': text,
            'sentiment': 'Positive' if pred == 1 else 'Negative',
            'confidence': prob[0][pred].item()
        }

def create_sentiment_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "darkgray"}
            ]
        },
        title={'text': "Confidence Score (%)"}
    ))
    fig.update_layout(height=250)
    return fig

def create_sentiment_distribution(results):
    sentiments = [r['sentiment'] for r in results]
    sentiment_counts = pd.Series(sentiments).value_counts()
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color_discrete_map={'Positive': 'green', 'Negative': 'red'}
    )
    return fig

def download_results(results):
    df = pd.DataFrame(results)
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sentiment_analysis_results_{timestamp}.csv"
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Results</a>'

def main():
    st.set_page_config(page_title="Nepali Sentiment Analysis", layout="wide")

    # Title and description
    st.title("नेपाली भावना विश्लेषण / Nepali Sentiment Analysis")
    st.markdown("""
    This application analyzes the sentiment of Nepali text using a fine-tuned BERT model.
    You can either enter text directly or upload a CSV file with multiple texts.
    """)

    # Initialize analyzer
    analyzer = NepaliSentimentAnalyzer()

    # Sidebar
    st.sidebar.title("Options")
    analysis_mode = st.sidebar.radio(
        "Choose Analysis Mode",
        ["Single Text Analysis", "Batch Analysis"]
    )

    if analysis_mode == "Single Text Analysis":
        # Single text analysis
        text_input = st.text_area(
            "Enter Nepali Text",
            height=100,
            placeholder="यहाँ नेपाली पाठ लेख्नुहोस्..."
        )

        if st.button("Analyze Sentiment"):
            if text_input.strip():
                with st.spinner("Analyzing sentiment..."):
                    result = analyzer.predict(text_input)

                # Display results in columns
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Analysis Results")
                    st.write(f"**Text:** {result['text']}")
                    st.write(f"**Sentiment:** {result['sentiment']}")
                    st.write(f"**Confidence:** {result['confidence']*100:.2f}%")

                with col2:
                    st.subheader("Confidence Gauge")
                    gauge_fig = create_sentiment_gauge(result['confidence'])
                    st.plotly_chart(gauge_fig, use_container_width=True)

                # Probability distribution
                st.subheader("Probability Distribution")
                print(result, result.keys())
                prob_fig = px.bar(
                    x=['Negative', 'Positive'],
                    y=[result['probabilities']['negative'], result['probabilities']['positive']],
                    title="Sentiment Probabilities"
                )
                st.plotly_chart(prob_fig, use_container_width=True)

            else:
                st.warning("Please enter some text to analyze.")

    else:
        # Batch analysis
        st.subheader("Batch Analysis")
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' not in df.columns:
                    st.error("CSV file must contain a 'text' column.")
                    st.stop()

                if st.button("Analyze Batch"):
                    with st.spinner("Analyzing batch of texts..."):
                        results = analyzer.analyze_batch(df['text'].tolist())

                    # Display results
                    st.subheader("Analysis Results")

                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["Summary", "Detailed Results", "Download"])

                    with tab1:
                        col1, col2 = st.columns(2)

                        with col1:
                            # Sentiment distribution pie chart
                            dist_fig = create_sentiment_distribution(results)
                            st.plotly_chart(dist_fig, use_container_width=True)

                        with col2:
                            # Summary statistics
                            total = len(results)
                            positive = sum(1 for r in results if r['sentiment'] == 'Positive')
                            negative = total - positive

                            st.metric("Total Texts Analyzed", total)
                            st.metric("Positive Sentiments", positive)
                            st.metric("Negative Sentiments", negative)

                    with tab2:
                        # Detailed results table
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df)

                    with tab3:
                        # Download link
                        st.markdown(download_results(results), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
    ### About
    This application uses a fine-tuned BERT model for Nepali sentiment analysis.
    The model was trained on a dataset of Nepali texts and their sentiment labels.
    """)

if __name__ == "__main__":
    main()