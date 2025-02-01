import sys
import threading
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton, QLabel,
    QVBoxLayout, QWidget, QProgressBar
)
from PyQt5.QtCore import pyqtSignal, QObject
from pypdf import PdfReader
import torch
from transformers import (
    pipeline, AutoTokenizer, AutoModelForQuestionAnswering
)
from transformers.utils import logging
from datasets import Dataset
import pandas as pd


class WorkerSignals(QObject):
    select_csv = pyqtSignal()
    select_jsonl = pyqtSignal()
    select_txt = pyqtSignal()
    update_progress = pyqtSignal(int)
    update_label = pyqtSignal(str)
    show_message = pyqtSignal(str)


class PDFQAApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

        # Initialize variables
        self.output_csv_path = None
        self.jsonl_file_path = None
        self.txt_file_path = None

    def initUI(self):
        self.setWindowTitle('PDF to Q&A Dataset')
        self.setGeometry(100, 100, 800, 200)

        layout = QVBoxLayout()

        self.label = QLabel('Select a PDF file to generate Q&A dataset.')
        layout.addWidget(self.label)

        self.button = QPushButton('Select PDF', self)
        self.button.clicked.connect(self.select_pdf)
        layout.addWidget(self.button)

        # Add progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def select_pdf(self):
        options = QFileDialog.Options()
        pdf_path, _ = QFileDialog.getOpenFileName(
            self, "Select PDF", "", "PDF Files (*.pdf);;All Files (*)", options=options)
        if pdf_path:
            # Start processing in a new thread
            self.thread = threading.Thread(target=self.generate_qa_dataset, args=(pdf_path,))
            self.thread.start()

    def generate_qa_dataset(self, pdf_path):
        # Create signals object
        self.signals = WorkerSignals()

        # Connect signals to slots
        self.signals.select_csv.connect(self.get_csv_save_path)
        self.signals.select_jsonl.connect(self.get_jsonl_save_path)
        self.signals.select_txt.connect(self.get_txt_save_path)
        self.signals.update_progress.connect(self.progress_bar.setValue)
        self.signals.update_label.connect(self.label.setText)
        self.signals.show_message.connect(self.show_message)

        # Extract text from PDF
        self.signals.update_label.emit('Extracting text from PDF...')
        text = ""
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            text += page_text + "\n"

        # Set model path to your local directory
        model_path = r"C:\Users\parse\OneDrive\Desktop\Ai-Studio\HuggingFace"

        # Load pre-trained model and tokenizer from local files
        self.signals.update_label.emit('Loading model from local files...')
        self.signals.update_progress.emit(0)
        logging.set_verbosity_error()  # Suppress warnings

        # Load model and tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        except Exception as e:
            self.signals.show_message.emit(f'Error loading model: {e}')
            return

        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

        self.signals.update_label.emit('Generating Q&A pairs...')
        qa_pairs = []

        # Split text into chunks and generate Q&A pairs
        chunk_size = 500
        total_chunks = len(text) // chunk_size + 1
        for idx, i in enumerate(range(0, len(text), chunk_size)):
            chunk = text[i:i+chunk_size]
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            start_index = torch.argmax(start_logits)
            end_index = torch.argmax(end_logits)
            input_ids = inputs["input_ids"][0]
            question_tokens = input_ids[start_index:end_index+1]
            question = tokenizer.decode(question_tokens, skip_special_tokens=True)
            if not question.strip():
                continue

            try:
                answer = qa_pipeline(question=question, context=chunk)['answer']
            except Exception as e:
                answer = "Could not generate answer."
            qa_pairs.append({"question": question, "answer": answer, "context": chunk})

            # Update progress bar
            progress = int((idx + 1) / total_chunks * 100)
            self.signals.update_progress.emit(progress)

        # Create dataset
        qa_dataset = Dataset.from_list(qa_pairs)

        # Select output CSV file
        self.signals.update_label.emit('Select location to save CSV file.')
        self.output_csv_path = None  # Reset the path
        self.signals.select_csv.emit()

        # Wait until the save path is set
        while self.output_csv_path is None:
            pass

        if self.output_csv_path:
            qa_dataset.to_csv(self.output_csv_path, index=False)

            # Convert to JSONL
            self.signals.update_label.emit('Select location to save JSONL file.')
            self.jsonl_file_path = None  # Reset the path
            self.signals.select_jsonl.emit()

            # Wait until the save path is set
            while self.jsonl_file_path is None:
                pass

            df = pd.read_csv(self.output_csv_path)
            if self.jsonl_file_path:
                df.to_json(self.jsonl_file_path, orient='records', lines=True)

            # Save as text file
            self.signals.update_label.emit('Select location to save Text file.')
            self.txt_file_path = None  # Reset the path
            self.signals.select_txt.emit()

            # Wait until the save path is set
            while self.txt_file_path is None:
                pass

            if self.txt_file_path:
                with open(self.txt_file_path, 'w', encoding='utf-8') as txt_file:
                    for idx, qa in enumerate(qa_pairs):
                        txt_file.write(f"Q{idx+1}: {qa['question']}\n")
                        txt_file.write(f"A{idx+1}: {qa['answer']}\n")
                        txt_file.write(f"Context: {qa['context']}\n")
                        txt_file.write("\n" + "="*50 + "\n\n")

                self.signals.update_label.emit(
                    f'Dataset saved as:\n{self.output_csv_path}\n{self.jsonl_file_path}\nand\n{self.txt_file_path}')
            else:
                self.signals.update_label.emit(
                    f'Dataset saved as:\n{self.output_csv_path}\nand\n{self.jsonl_file_path}')
        else:
            self.signals.update_label.emit('Dataset saving cancelled.')

        self.signals.update_progress.emit(0)  # Reset progress bar

    def get_csv_save_path(self):
        options = QFileDialog.Options()
        output_csv_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        self.output_csv_path = output_csv_path

    def get_jsonl_save_path(self):
        options = QFileDialog.Options()
        jsonl_file_path, _ = QFileDialog.getSaveFileName(
            self, "Save JSONL File", "", "JSONL Files (*.jsonl);;All Files (*)", options=options)
        self.jsonl_file_path = jsonl_file_path

    def get_txt_save_path(self):
        options = QFileDialog.Options()
        txt_file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Text File", "", "Text Files (*.txt);;All Files (*)", options=options)
        self.txt_file_path = txt_file_path

    def show_message(self, message):
        self.label.setText(message)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PDFQAApp()
    ex.show()
    sys.exit(app.exec_())
