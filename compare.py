import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog
)
import re

class ModelComparator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Output Comparator")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Reference file picker
        ref_layout = QHBoxLayout()
        ref_label = QLabel("Reference JSON:")
        self.ref_edit = QLineEdit()
        self.ref_edit.setReadOnly(True)
        ref_btn = QPushButton("Browse")
        ref_btn.clicked.connect(lambda: self.load_file(self.ref_edit, "Reference"))
        ref_layout.addWidget(ref_label)
        ref_layout.addWidget(self.ref_edit)
        ref_layout.addWidget(ref_btn)
        layout.addLayout(ref_layout)
        
        # Model output file pickers
        self.model_pickers = []
        for i in range(6):
            picker_layout = QHBoxLayout()
            picker_label = QLabel(f"Model {i+1} Output:")
            picker_edit = QLineEdit()
            picker_edit.setReadOnly(True)
            picker_btn = QPushButton("Browse")
            picker_btn.clicked.connect(lambda checked, edit=picker_edit, num=i+1: 
                                    self.load_file(edit, f"Model {num}"))
            picker_layout.addWidget(picker_label)
            picker_layout.addWidget(picker_edit)
            picker_layout.addWidget(picker_btn)
            layout.addLayout(picker_layout)
            self.model_pickers.append(picker_edit)
        
        # Compare button
        compare_btn = QPushButton("Compare Outputs")
        compare_btn.clicked.connect(self.compare_outputs)
        layout.addWidget(compare_btn)
        
        # Results display
        results_label = QLabel("Comparison Results:")
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(results_label)
        layout.addWidget(self.results_text)

    def load_file(self, edit_field, source):
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {source} File",
            os.getcwd(),
            "All Files (*)"
        )
        if path:
            edit_field.setText(path)

    def extract_parts(self, content):
        parts = []
        part_matches = re.finditer(r'["\s](\d{7}|\d{8})["\s]', content)
        part_name_pattern = r'["\s]([^"]*?)"?\s*,?\s*(?:part_number|"part_number")\s*:\s*"?\d{7,8}"?'
        
        for match in part_matches:
            part_number = match.group(1)
            content_before = content[:match.start()]
            name_match = re.findall(part_name_pattern, content_before, re.IGNORECASE)
            if name_match:
                part_name = name_match[-1].strip().strip('"').strip()
                parts.append({
                    "part_number": part_number,
                    "part_name": part_name
                })
        return parts

    def load_json_from_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                return self.extract_parts(content)
        except Exception as e:
            self.results_text.append(f"Error loading {filepath}: {str(e)}")
            return None

    def normalize_part_data(self, parts):
        normalized = {}
        for part in parts:
            if isinstance(part, dict) and 'part_number' in part and 'part_name' in part:
                number = part['part_number'].strip()
                # Split name into significant words, ignoring common words and formatting
                name = part['part_name'].strip().lower()
                # Remove special characters and extra spaces
                name = re.sub(r'[^\w\s]', ' ', name)
                # Split into words and remove common words/numbers
                words = set(w for w in name.split() if len(w) > 2 and not w.isdigit())
                normalized[number] = words
        return normalized

    def calculate_name_similarity(self, ref_words, model_words):
        if not ref_words or not model_words:
            return 0
        common_words = ref_words.intersection(model_words)
        return len(common_words) / len(ref_words) >= 0.5  # At least 50% word match

    def calculate_scores(self, reference, model_output):
        if not reference or not model_output:
            return 0, 0, 0, "N/A"

        ref_parts = set(reference.keys())
        model_parts = set(model_output.keys())
        
        correct_numbers = ref_parts.intersection(model_parts)
        correct_names = 0
        mismatches = []
        matches = []
        
        for num in correct_numbers:
            if self.calculate_name_similarity(reference[num], model_output[num]):
                correct_names += 1
                matches.append((num, reference[num], model_output[num]))
            else:
                mismatches.append((num, reference[num], model_output[num]))
        
        part_number_accuracy = len(correct_numbers) / len(ref_parts) * 100
        name_accuracy = correct_names / len(ref_parts) * 100
        overall_score = (part_number_accuracy + name_accuracy) / 2
        
        details = (f"Part Numbers: {len(correct_numbers)}/{len(ref_parts)} "
                  f"({part_number_accuracy:.1f}%)\n"
                  f"Correct Names: {correct_names}/{len(ref_parts)} "
                  f"({name_accuracy:.1f}%)\n\n"
                  f"Sample Mismatches (first 3):\n" +
                  '\n'.join([f"Part {num}: Ref={ref} vs Model={model}" 
                            for num, ref, model in mismatches[:3]]))
        
        return part_number_accuracy, name_accuracy, overall_score, details

    def compare_outputs(self):
        if not self.ref_edit.text():
            self.results_text.setText("Please select a reference file first.")
            return
            
        reference_data = self.load_json_from_file(self.ref_edit.text())
        if not reference_data:
            return
            
        reference = self.normalize_part_data(reference_data)
        self.results_text.clear()
        self.results_text.append(f"Reference file contains {len(reference)} unique parts\n")
        
        for i, picker in enumerate(self.model_pickers):
            if not picker.text():
                continue
                
            model_data = self.load_json_from_file(picker.text())
            if not model_data:
                continue
                
            model_parts = self.normalize_part_data(model_data)
            part_acc, name_acc, overall, details = self.calculate_scores(
                reference, model_parts
            )
            
            self.results_text.append(f"\nModel {i+1} Results:")
            self.results_text.append(f"Total parts found: {len(model_parts)}")
            self.results_text.append(f"Overall Score: {overall:.1f}%")
            self.results_text.append(details)
            self.results_text.append("-" * 40)

def main():
    app = QApplication(sys.argv)
    window = ModelComparator()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()