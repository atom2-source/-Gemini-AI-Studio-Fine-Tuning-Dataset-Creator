import sys
import os
import base64
import json
import requests

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QFile, QIODevice
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QProgressBar
)
from PyQt5.QtGui import QImage

CHUNK_SIZE = 15  # how many PDF pages per chunk

class DocAIWorker(QThread):
    """
    Processes a single PDF:
      1) Load PDF file as binary
      2) Document AI process
      3) Save results and extract images
    """

    progress_msg = pyqtSignal(str)
    progress_val = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(
        self,
        pdf_path,
        output_dir,
        project_number,
        processor_id,
        location
    ):
        super().__init__()
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.project_number = project_number
        self.processor_id = processor_id
        self.location = location

    def run(self):
        try:
            pdf_basename = os.path.splitext(os.path.basename(self.pdf_path))[0]
            
            # Prepare output directory
            docai_dir = os.path.join(self.output_dir, "docai_results")
            os.makedirs(docai_dir, exist_ok=True)

            images_dir = os.path.join(self.output_dir, "docai_images")
            os.makedirs(images_dir, exist_ok=True)

            # 1) Get access token
            token = self.get_access_token()

            # 2) Process PDF with Document AI
            self.update_progress_msg("Processing PDF with Document AI...")
            docai_resp = self.process_pdf(self.pdf_path, token)
            
            # Update progress
            self.progress_val.emit(50)

            # 3) Save results
            combined_json_path = self.save_json_result(docai_resp, pdf_basename, docai_dir)
            combined_txt_path = self.save_text_result(docai_resp, pdf_basename, docai_dir)

            # 4) Extract images
            self.update_progress_msg("Extracting images...")
            self.extract_images(docai_resp, images_dir, pdf_basename)
            self.progress_val.emit(100)

            # 5) Summarize
            done_msg = (
                f"All done.\n"
                f"Document AI => {combined_json_path}\n"
                f"docAI text => {combined_txt_path}\n"
                f"Images => {images_dir}\n"
            )
            self.finished.emit(done_msg)

        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")

    def get_access_token(self):
        self.update_progress_msg("Fetching GCP access token via gcloud...")
        token = os.popen("gcloud auth print-access-token").read().strip()
        if not token:
            raise RuntimeError("No valid access token found. Please run 'gcloud auth login'.")
        return token

    def process_pdf(self, pdf_path, token):
        self.update_progress_msg(f"Reading PDF file: {os.path.basename(pdf_path)}")
        
        # Read PDF file using QFile
        pdf_file = QFile(pdf_path)
        if not pdf_file.open(QIODevice.ReadOnly):
            raise RuntimeError(f"Cannot read PDF file: {pdf_path}")
        
        file_bytes = pdf_file.readAll().data()
        pdf_file.close()
        
        encoded = base64.b64encode(file_bytes).decode("utf-8")

        url = (
            f"https://{self.location}-documentai.googleapis.com/v1/"
            f"projects/{self.project_number}/locations/{self.location}/processors/"
            f"{self.processor_id}:process"
        )
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8"
        }
        payload = {
            "rawDocument": {
                "mimeType": "application/pdf",
                "content": encoded
            },
            "processOptions": {
                "ocrConfig": {
                    "enableNativePdfParsing": True,
                    "enableImageQualityScores": True
                }
            }
        }

        r = requests.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

    def save_json_result(self, docai_resp, base_name, docai_dir):
        out_json = os.path.join(docai_dir, f"{base_name}_result.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(docai_resp, f, indent=2)
        self.update_progress_msg(f"Saved docAI JSON => {out_json}")
        return out_json

    def save_text_result(self, docai_resp, base_name, docai_dir):
        out_txt = os.path.join(docai_dir, f"{base_name}_result.txt")
        text = docai_resp.get("document", {}).get("text", "")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(text)
        self.update_progress_msg(f"Saved docAI text => {out_txt}")
        return out_txt

    def extract_images(self, docai_resp, images_dir, base_name):
        doc_obj = docai_resp.get("document", {})
        pages = doc_obj.get("pages", [])
        
        total_imgs = 0
        for page_num, page in enumerate(pages, start=1):
            img_info = page.get("image", {})
            b64_img = img_info.get("content")
            mime = img_info.get("mimeType", "image/png")
            
            if b64_img:
                content = base64.b64decode(b64_img)
                if "png" in mime.lower():
                    ext = ".png"
                elif "jpg" in mime.lower() or "jpeg" in mime.lower():
                    ext = ".jpg"
                else:
                    ext = ".bin"

                fname = f"{base_name}_page{page_num}{ext}"
                out_path = os.path.join(images_dir, fname)
                with open(out_path, "wb") as out:
                    out.write(content)
                total_imgs += 1
                self.update_progress_msg(f"  Extracted image => {fname}")

        self.update_progress_msg(f"Done extracting {total_imgs} images.")

    def update_progress_msg(self, msg):
        self.progress_msg.emit(msg)


class DocAIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Document AI Extraction")
        self.setGeometry(100, 100, 900, 600)

        cw = QWidget()
        self.setCentralWidget(cw)
        main_layout = QVBoxLayout(cw)

        self.status_label = QLabel("Status: Ready")
        main_layout.addWidget(self.status_label)

        # PDF input
        pdf_layout = QHBoxLayout()
        pdf_layout.addWidget(QLabel("PDF File:"))
        self.pdf_edit = QLineEdit()
        pdf_layout.addWidget(self.pdf_edit)
        browse_pdf_btn = QPushButton("Browse PDF")
        browse_pdf_btn.clicked.connect(self.pick_pdf)
        pdf_layout.addWidget(browse_pdf_btn)
        main_layout.addLayout(pdf_layout)

        # Output
        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("Output Folder:"))
        self.out_edit = QLineEdit()
        out_layout.addWidget(self.out_edit)
        browse_out_btn = QPushButton("Browse Folder")
        browse_out_btn.clicked.connect(self.pick_out)
        out_layout.addWidget(browse_out_btn)
        main_layout.addLayout(out_layout)

        # Start
        self.start_btn = QPushButton("Run Document AI")
        self.start_btn.clicked.connect(self.on_start)
        main_layout.addWidget(self.start_btn)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # Log
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        main_layout.addWidget(self.log_area)

        # Document AI config
        self.project_number = "44469913499"
        self.processor_id = "a06930975e13235"
        self.location = "us"

        self.worker = None

    def pick_pdf(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select PDF", os.getcwd(), "PDF Files (*.pdf)"
        )
        if f:
            self.pdf_edit.setText(f)

    def pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select Folder", os.getcwd())
        if d:
            self.out_edit.setText(d)

    def on_start(self):
        pdf_path = self.pdf_edit.text().strip()
        out_dir = self.out_edit.text().strip()
        if not pdf_path or not os.path.isfile(pdf_path):
            self.update_status("Error: Invalid PDF file.")
            return
        if not out_dir or not os.path.isdir(out_dir):
            self.update_status("Error: Invalid output folder.")
            return

        self.log_area.clear()
        self.progress_bar.setValue(0)
        self.update_status("Starting Document AI extraction...")

        self.start_btn.setEnabled(False)

        self.worker = DocAIWorker(
            pdf_path,
            out_dir,
            self.project_number,
            self.processor_id,
            self.location
        )
        self.worker.progress_msg.connect(self.on_worker_msg)
        self.worker.progress_val.connect(self.on_worker_val)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_msg(self, msg):
        self.update_status(msg)

    def on_worker_val(self, v):
        self.progress_bar.setValue(v)

    def on_worker_finished(self, result):
        self.start_btn.setEnabled(True)
        self.update_status(result)
        self.worker = None

    def update_status(self, msg):
        self.status_label.setText(f"Status: {msg}")
        self.log_area.append(msg)


def main():
    app = QApplication(sys.argv)
    w = DocAIApp()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()