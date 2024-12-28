# DisneyLand-review-analysis
This project analyzes sentiment in reviews about Disneyland, providing insights into customer experiences and areas for improvement. Using natural language processing (NLP) techniques, the project categorizes reviews into positive, neutral, or negative sentiment.
## Features
- **Sentiment Classification**: Classifies reviews into positive, negative, or neutral categories.
- **NLP Preprocessing**: Tokenization, stemming, stopword removal, and more.
- **Data Visualization**: Word clouds, sentiment trends, and other visual analytics.
- **Machine Learning Models**: Uses classifiers such as Logistic Regression, Random Forest, and advanced transformers like BERT.

---

## Project Structure
```
├── data/                 # Directory for storing raw and processed datasets
├── notebooks/            # Jupyter notebooks for experiments and analysis
│   └── DisneylandReviewAnalysis.ipynb
├── src/                  # Core scripts for preprocessing, training, and evaluation
├── models/               # Pre-trained and custom model files
├── results/              # Output sentiment scores, visualizations, and metrics
├── README.md             # Project documentation
└── requirements.txt      # Required Python libraries
```

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Libraries listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/pridzzy/DisneyLand-review-analysis.git
   cd disneyland-reviews-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   - Place the Disneyland review dataset in the `data/` directory.
   - Ensure the file format is `.csv` or update preprocessing scripts to handle your format.

---

## Usage

### Preprocessing
Prepare the data for analysis:
```bash
python src/preprocess.py
```

### Model Training
Train the sentiment classification model:
```bash
python src/train.py --config configs/train_config.json
```

### Evaluation
Evaluate the model and generate metrics:
```bash
python src/evaluate.py --model_path models/sentiment_model.pth
```

### Visualization
Generate insights from the review data:
```bash
python src/visualize.py
```

---

## Dataset
The project utilizes a dataset of Disneyland reviews, which includes:
- **Text Reviews**: Customer feedback.
- **Metadata**: Ratings, review date, and user information (optional).

---

## Results
- **Accuracy**: `X%`
- **F1 Score**: `Y`
- **Sample Visualization**:
  - Word Clouds for most frequent terms.
  - Sentiment distribution pie charts.

---

## Future Enhancements
- Support for multilingual reviews.
- Real-time sentiment analysis for new reviews.
- Sentiment trend analysis over time.

---

## Contributing
We welcome contributions! Please:
1. Fork the repository.
2. Create a feature branch (`feature/your-feature`).
3. Commit changes and push the branch.
4. Open a pull request for review.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- Data source: Publicly available Disneyland review datasets.
- Libraries: Scikit-learn, TensorFlow, Hugging Face Transformers.

___
