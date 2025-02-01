Gemini AI Studio Fine-Tuning Dataset Creator
A PyQt5-based GUI application designed to generate structured fine-tuning datasets in the Dolly format using Google Gemini AI. This tool is particularly useful for creating high-quality training data from technical manuals or any subject-specific content.
Features

Dolly Format Dataset Generation: Creates training data following the Dolly instruction format, which includes instruction, context, response, and category fields
Bulk Context & Category Labeling: Add consistent context and category labels across all entries for efficient dataset organization
Flexible Subject Support: While designed for technical manuals, it works with any subject matter - just set the "machine name" to your topic of interest (e.g., "green tree frogs", "quantum physics")
Reference Q&A Integration: Input previous Q&A pairs to avoid duplicates and guide new question generation
Dark Mode Interface: Material Design-inspired dark theme for comfortable extended use
Multiple Gemini Models: Support for various Gemini AI models with real-time rate limit tracking
File Management: Import source text from files and export structured datasets in JSON or text format

Rate Limiting & API Usage
The application includes real-time tracking of API usage with built-in rate limiting for different Gemini models. This helps prevent hitting API limits while working with larger datasets.
Installation & Setup

Clone the repository
Install required packages![Screenshot 2025-02-02 013116](https://github.com/user-attachments/assets/8cb04008-a886-417b-b906-b4688aaa0756)

Set your GEMINI_API_KEY environment variable
Run the application

Dataset Format
The tool now generates data in the Dolly instruction format:
![Screenshot 2025-02-02 013038](https://github.com/user-attachments/assets/aa07fe50-8028-419e-931b-a47a4725ebe1)
