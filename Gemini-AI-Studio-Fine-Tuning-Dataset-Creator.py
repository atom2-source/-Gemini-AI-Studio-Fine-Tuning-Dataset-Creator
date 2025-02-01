"""
Instructions for setting the API key:

Before running this application, you must set the GEMINI_API_KEY environment variable.
For example, in your terminal (CLI), you can set the key as follows:

On Linux or macOS:
    export GEMINI_API_KEY="your_api_key_here"

On Windows (Command Prompt):
    set GEMINI_API_KEY="your_api_key_here"

Alternatively, you can set the environment variable in your IDE's run configuration.
"""

import sys
import os
import json
import time
from pathlib import Path
import logging
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QComboBox,
    QMessageBox
)
import google.generativeai as genai

class RateLimiter:
    MODELS = {
        'gemini-1.5-flash': {'rpm': 15, 'daily': 1500},
        'gemini-1.5-flash-8b': {'rpm': 15, 'daily': 1500},
        'gemini-1.5-pro': {'rpm': 2, 'daily': 50},
        'gemini-2.0-flash': {'rpm': 10, 'daily': 1500},
        'gemini-2.0-flash-thinking': {'rpm': 10, 'daily': 1500}
    }
    
    def __init__(self):
        self.log_file = Path('rate_limits.log')
        self.state_file = Path('rate_state.json')
        self.setup_logging()
        self.load_state()
        
    def setup_logging(self):
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
    def load_state(self):
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
        else:
            self.state = self._initialize_state()
            self.save_state()
            
    def _initialize_state(self):
        return {
            model: {
                'minute_start': time.time(),
                'day_start': time.time(),
                'minute_requests': 0,
                'daily_requests': 0
            }
            for model in self.MODELS
        }
        
    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f)
            
    def log_request(self, model):
        # Advisory only; never block the call.
        if model not in self.MODELS:
            logging.error(f"Unknown model: {model}")
            return True
            
        current_time = time.time()
        model_state = self.state[model]
        limits = self.MODELS[model]
        
        if current_time - model_state['minute_start'] >= 60:
            model_state['minute_start'] = current_time
            model_state['minute_requests'] = 0
            
        if current_time - model_state['day_start'] >= 86400:
            model_state['day_start'] = current_time
            model_state['daily_requests'] = 0
            
        if (model_state['minute_requests'] >= limits['rpm'] or 
            model_state['daily_requests'] >= limits['daily']):
            logging.warning(
                f"{model} - Rate limit advisory: "
                f"{limits['rpm'] - model_state['minute_requests']} requests remaining this minute, "
                f"{limits['daily'] - model_state['daily_requests']} requests remaining today."
            )
        model_state['minute_requests'] += 1
        model_state['daily_requests'] += 1
        remaining_minute = limits['rpm'] - model_state['minute_requests']
        remaining_daily = limits['daily'] - model_state['daily_requests']
        logging.info(
            f"{model} - Request logged. Remaining: {remaining_minute}/min, {remaining_daily}/day"
        )
        self.save_state()
        return True  # Always allow the call

    def get_limits(self):
        current_time = time.time()
        status = {}
        for model, state in self.state.items():
            limits = self.MODELS[model]
            remaining_minute = limits['rpm'] - state['minute_requests']
            remaining_daily = limits['daily'] - state['daily_requests']
            
            if current_time - state['minute_start'] >= 60:
                remaining_minute = limits['rpm']
            if current_time - state['day_start'] >= 86400:
                remaining_daily = limits['daily']
                
            status[model] = {
                'remaining_minute': remaining_minute,
                'remaining_daily': remaining_daily
            }
        return status

class GeminiOneShotWorker(QThread):
    progress_msg = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, text, model_name, generation_config, machine_name, additional_qa, parent=None):
        super().__init__(parent)
        self.text = text
        self.model_name = model_name
        self.generation_config = generation_config
        self.machine_name = machine_name
        self.additional_qa = additional_qa

    def run(self):
        try:
            self.update_log(f"Initializing Gemini model: {self.model_name}")
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config
            )
            chat_session = model.start_chat(history=[])
            prompt = self.build_prompt(self.text, self.machine_name, self.additional_qa)
            self.update_log("Sending manual to Gemini...")
            response = chat_session.send_message(prompt)
            raw_text = response.text or ""
            
            self.update_log(f"Raw response received: {raw_text[:500]}...")
            parsed_response = self.parse_response(raw_text)
            
            if isinstance(parsed_response, (dict, list)):
                output = json.dumps(parsed_response, indent=2)
            else:
                output = str(parsed_response)
            self.finished.emit(output)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.update_log(error_msg)
            self.finished.emit(error_msg)

    def build_prompt(self, input_text, machine_name, additional_qa):
        # Few-shot example with a sample manual excerpt.
        few_shot_examples = (
            "Example Manual Excerpt:\n"
            "Machine: CNC Lathe\n"
            "Part: Main Spindle, part number 1234567, operates at 2000 RPM, requires lubrication every 500 hours.\n\n"
            "Expected Output (Fine-Tuning Dataset Entry):\n"
            "[\n"
            "  {\n"
            '    "question": "For CNC Lathe, what is the part number for the Main Spindle?",\n'
            '    "answer": "1234567"\n'
            "  },\n"
            "  {\n"
            '    "question": "For CNC Lathe, what are the operational details of the Main Spindle?",\n'
            '    "answer": "Operates at 2000 RPM and requires lubrication every 500 hours."\n'
            "  }\n"
            "]\n"
        )
        
        qa_prompt = ""
        if additional_qa:
            qa_prompt = (
                "The user provided the following Q&A examples:\n"
                f"{additional_qa}\n\n"
                "Based on these examples, generate different questions and answers, and also review the manual for any parts or identification numbers not covered. "
                "Produce additional fine-tuning dataset entries as needed.\n\n"
            )
        
        prompt = (
            "You are an expert in comprehending technical manuals and creating fine-tuning datasets. Your task is to extract every detail from the manual provided below. "
            "For each part mentioned, generate at least one question that includes the machine name (as specified) and asks for the part number, along with additional questions covering operational or descriptive details. "
            "Ensure that every question includes the machine name and is answered with the correct information. "
            "Your output should be a JSON array where each element represents a fine-tuning data point with the keys 'question' and 'answer'. "
            "Exhaust all available details.\n\n"
            f"{few_shot_examples}\n"
            f"{qa_prompt}"
            f"Machine Name: {machine_name}\n"
            "Manual Text:\n"
            f"{input_text}"
        )
        return prompt

    def parse_response(self, raw_text):
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return raw_text

    def update_log(self, msg):
        self.progress_msg.emit(msg)

class GeminiTesterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Updated window title to reflect fine-tuning dataset creation
        self.setWindowTitle("Gemini AI Studio Fine-Tuning Dataset Creator")
        self.setGeometry(100, 100, 800, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Model selection layout
        model_layout = QHBoxLayout()
        model_label = QLabel("Select Gemini Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "gemini-exp-1206",
            "gemini-2.0-flash-exp-01-21",
            "gemini-2.0-flash-thinking-exp-01-21",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b"
        ])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        self.model_combo.currentIndexChanged.connect(self.update_limits_display)
        layout.addLayout(model_layout)

        # Machine name input layout
        machine_layout = QHBoxLayout()
        machine_label = QLabel("Machine Name:")
        self.machine_edit = QLineEdit()
        machine_layout.addWidget(machine_label)
        machine_layout.addWidget(self.machine_edit)
        layout.addLayout(machine_layout)

        # Additional Q&A input layout (optional)
        qa_layout = QVBoxLayout()
        qa_label = QLabel("Additional Q&A (optional):")
        self.qa_edit = QTextEdit()
        self.qa_edit.setPlaceholderText("Enter any example Q&A pairs here...")
        qa_layout.addWidget(qa_label)
        qa_layout.addWidget(self.qa_edit)
        layout.addLayout(qa_layout)

        # Input manual text layout
        input_layout = QHBoxLayout()
        input_label = QLabel("Input Manual Text:")
        self.input_edit = QTextEdit()
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_edit)
        layout.addLayout(input_layout)

        # File load layout
        load_layout = QHBoxLayout()
        load_label = QLabel("Or Load Text from File:")
        self.load_edit = QLineEdit()
        self.load_edit.setReadOnly(True)
        load_btn = QPushButton("Browse")
        load_btn.clicked.connect(self.load_text_from_file)
        load_layout.addWidget(load_label)
        load_layout.addWidget(self.load_edit)
        load_layout.addWidget(load_btn)
        layout.addLayout(load_layout)

        # Run button (updated text)
        self.run_btn = QPushButton("Create Fine-Tuning Dataset")
        self.run_btn.clicked.connect(self.run_gemini)
        layout.addWidget(self.run_btn)

        # Output display layout (updated label)
        output_layout = QVBoxLayout()
        output_label = QLabel("Fine-Tuning Dataset Output:")
        self.output_edit = QTextEdit()
        self.output_edit.setReadOnly(True)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_edit)
        layout.addLayout(output_layout)

        # Save response layout
        save_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Response")
        self.save_btn.clicked.connect(self.save_response)
        self.save_btn.setEnabled(False)
        save_layout.addStretch()
        save_layout.addWidget(self.save_btn)
        layout.addLayout(save_layout)

        # Log display layout
        log_layout = QVBoxLayout()
        log_label = QLabel("Logs:")
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        log_layout.addWidget(log_label)
        log_layout.addWidget(self.log_edit)
        layout.addLayout(log_layout)

        self.init_gemini_api()
        self.init_rate_limiter()
        self.worker = None

    def init_gemini_api(self):
        try:
            api_key = os.environ["GEMINI_API_KEY"]
            genai.configure(api_key=api_key)
            self.log("Gemini API configured successfully.")
        except KeyError:
            self.log("Error: GEMINI_API_KEY environment variable not set.")
            QMessageBox.critical(self, "API Key Error", 
                                 "Please set the GEMINI_API_KEY environment variable.")
            sys.exit(1)

    def init_rate_limiter(self):
        self.rate_limiter = RateLimiter()
        self.update_limits_display()

    def update_limits_display(self):
        limits = self.rate_limiter.get_limits()
        current_model = self.model_combo.currentText()
        if current_model in limits:
            status = limits[current_model]
            self.log(f"Current limits for {current_model}:")
            self.log(f"Remaining requests per minute: {status['remaining_minute']}")
            self.log(f"Remaining requests per day: {status['remaining_daily']}")
        else:
            self.log(f"No rate limit information available for {current_model}")

    def load_text_from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Text File",
            os.getcwd(),
            "Text Files (*.txt);;All Files (*)"
        )
        if path:
            self.load_edit.setText(path)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                self.input_edit.setText(text)
                self.log(f"Loaded text from {path}")
            except Exception as e:
                self.log(f"Failed to load text: {str(e)}")
                QMessageBox.warning(self, "Load Error", f"Failed to load text: {str(e)}")

    def run_gemini(self):
        model_name = self.model_combo.currentText()
        input_text = self.input_edit.toPlainText().strip()
        machine_name = self.machine_edit.text().strip()
        additional_qa = self.qa_edit.toPlainText().strip()
        
        if not machine_name:
            self.log("Error: Machine name is empty.")
            QMessageBox.warning(self, "Input Error", "Please enter a machine name.")
            return

        if not input_text:
            self.log("Error: Input text is empty.")
            QMessageBox.warning(self, "Input Error", "Please enter or load some manual text to process.")
            return

        # Log the request advisory (does not block the call)
        self.rate_limiter.log_request(model_name)

        if model_name == "gemini-1.5-flash-8b":
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }
        else:
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 65536,
                "response_mime_type": "text/plain",
            }

        self.run_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.output_edit.clear()

        self.worker = GeminiOneShotWorker(input_text, model_name, generation_config, machine_name, additional_qa)
        self.worker.progress_msg.connect(self.log)
        self.worker.finished.connect(self.display_response)
        self.worker.start()

    def display_response(self, response_text):
        self.output_edit.setPlainText(response_text)
        self.run_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.log("Fine-tuning dataset creation completed.")

    def save_response(self):
        response_text = self.output_edit.toPlainText().strip()
        if not response_text:
            self.log("Error: No response to save.")
            QMessageBox.warning(self, "Save Error", "There is no response to save.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Response",
            os.getcwd(),
            "Text Files (*.txt);;JSON Files (*.json);;All Files (*)"
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(response_text)
                self.log(f"Response saved to {path}")
                QMessageBox.information(self, "Save Successful", f"Response saved to {path}")
            except Exception as e:
                self.log(f"Failed to save response: {str(e)}")
                QMessageBox.warning(self, "Save Error", f"Failed to save response: {str(e)}")

    def log(self, message):
        self.log_edit.append(message)

def main():
    app = QApplication(sys.argv)
    # Dark theme inspired by Material design for dark mode
    app.setStyleSheet("""
        QWidget {
            font-family: "Roboto", "Helvetica Neue", sans-serif;
            font-size: 14px;
            background-color: #121212;
            color: #E0E0E0;
        }
        QMainWindow {
            background-color: #121212;
        }
        QLineEdit, QTextEdit, QComboBox {
            background-color: #1E1E1E;
            border: 1px solid #444444;
            color: #E0E0E0;
            border-radius: 4px;
            padding: 6px;
        }
        QPushButton {
            background-color: #BB86FC;
            color: #121212;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
        }
        QPushButton:hover {
            background-color: #985EFF;
        }
        QPushButton:pressed {
            background-color: #BB86FC;
        }
        QLabel {
            color: #E0E0E0;
        }
    """)
    window = GeminiTesterApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
