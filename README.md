# InterviewGenie

## 🚀 Overview
InterviewGenie is an AI-powered interview preparation tool that leverages Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG) to simulate realistic interview scenarios. The tool supports two modes:

1. **Interviewee Mode**: The AI acts as the interviewee, generating intelligent answers to user-provided questions using the `google/gemma-2b-it` model.
2. **Interviewer Mode**: The AI acts as the interviewer, generating questions and evaluating responses with two different models:
   - **Free Model**: Generates `n` questions using `google/gemma-2b-it`, asks the user to answer them, and displays the complete question-answer set.
   - **Paid Model**: Uses the OpenAI API to generate questions, collects user responses, and provides a detailed evaluation with a score and custom feedback for each answer to enhance interview performance.

### 🔥 Why Use InterviewGenie?
This project streamlines the interview process by simulating an interviewer while also assisting users in preparing for interviews. It enhances efficiency, accelerates preparation, and helps users progressively improve their performance with each iteration of the model.

---

## 🛠️ Features
- AI-powered **question-answer generation** using `google/gemma-2b-it`.
- **Two modes**: Interviewer and Interviewee.
- **Free model** for basic question generation and answer recording.
- **Paid model** with enhanced OpenAI-based question generation and personalized feedback.
- **Custom scoring system** for detailed performance evaluation.
- **Logging and Exception Handling** for better debugging and reliability.

---

## 💂‍♂️ Project Structure
```
InterviewGenie End-To-End
│── artifacts/
│── data/
│── src/
│   ├── components/
│   │   ├── data_ingestion.py  # Handles data collection and storage
│   │   ├── data_transformation.py  # Processes and transforms raw data
│   │   ├── model_trainer.py  # Trains and optimizes models
│   ├── helper/
│   │   ├── interviewee.py  # Handles responses for the interviewee mode
│   │   ├── interviewer.py  # Generates questions and evaluates responses
│   │   ├── question_generation.py  # Implements the question generation logic
│   ├── pipeline/
│   │   ├── predict_pipeline.py  # Handles model predictions
│   │   ├── train_pipeline.py  # Automates the model training process
│   ├── exception.py  # Custom exception handling
│   ├── logger.py  # Logging setup for tracking events and errors
│   ├── utils.py  # Utility functions for supporting operations
│── templates/
│   ├── base.html  # Base template for consistent UI
│   ├── index.html  # Homepage UI
│   ├── interviewee.html  # UI for interviewee interactions
│   ├── interviewer.html  # UI for interviewer interactions
│   ├── question_sheet.html  # Displays generated questions
│   ├── result.html  # Displays final scores and evaluations
│── static/  # Stores static assets like CSS and JavaScript
│── logs/  # Stores application logs
│── app.py  # Main Flask application
│── setup.py  # Project setup configuration
│── requirements.txt  # Dependencies
│── README.md  # Documentation
```

---

## 🛠️ Installation & Setup
### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Installation
1. **Clone the repository**
   ```sh
   git clone https://github.com/tanaydwivedi095/InterviewGenie_v2.git
   cd InterviewGenie_v2
   ```
2. **Create a virtual environment (optional but recommended)**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```
3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

---

## 🚀 Usage
1. **Run the Flask app**
   ```sh
   python app.py
   ```
2. **Open the web interface** in a browser:
   ```
   http://127.0.0.1:8080/
   ```
3. **Select a mode:**
   - Interviewee mode: Enter a question, and the AI generates an answer.
   - Interviewer mode:
     - **Free Model**: AI generates questions, user answers, and responses are displayed.
     - **Paid Model**: AI generates questions using OpenAI, collects user answers, scores them, and provides personalized feedback.

---

## 📌 How to Create a RAG Pipeline
If you want to create your own **Retrieval-Augmented Generation (RAG) pipeline**, follow these steps:

1. **Create a `data/` folder under `src/components/`.**
   ```sh
   mkdir -p src/components/data
   ```
2. **Dump all the PDF files** that you want to use for training inside `src/components/data/`.
3. **Run `train_pipeline.py`** to process and train the model.
   ```sh
   python src/pipeline/train_pipeline.py
   ```

⚠️ **Note**: Running this step is necessary when using the model for the first time.

This will ensure that the data is ingested and transformed properly for the RAG-based interview preparation system.

---

## 🤖 Technologies Used
- **Python**
- **HuggingFace Transformers** (`google/gemma-2b-it`)
- **OpenAI API** (for the paid model)
- **PyTorch** (for model processing)
- **Flask** (for the web interface)
- **HTML, Jinja, CSS** (for frontend rendering)
- **Logging Module** (for error tracking)

---

## 🎯 Future Improvements
- **Integration with more LLMs** for enhanced interview question generation.
- **Speech-based interaction** for voice interviews.
- **Resume analysis** for personalized question sets.
- **Improved logging and debugging tools**.

---

## 🤝 Contribution
Contributions are welcome! Feel free to submit pull requests or report issues.

---

## 🐝 License
MIT License. See `LICENSE` for details.

---

## 💎 Contact
For any queries or issues, reach out at [tanaydwivedi095@gmail.com](mailto:tanaydwivedi095@gmail.com).

