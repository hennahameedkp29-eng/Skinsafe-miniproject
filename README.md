# SkinSafe

SkinSafe is a Flask-based web application that helps users discover skincare products suited to their skin type and concerns. It combines a personalized skin assessment quiz with an SBERT + FAISS-powered recommendation engine to surface relevant products from a curated catalog.

## Features

- **User accounts** — sign up, log in, and manage a personal profile
- **Skin assessment quiz** — a guided test to identify skin type and concerns
- **Personalized recommendations** — semantic product matching using SBERT embeddings and FAISS similarity search
- **Product browsing** — explore products by category, search by name/concern, and view detailed product pages
- **Saved products** — bookmark products to revisit later

## Tech Stack

- **Backend:** Python, Flask
- **Recommendation Engine:** Sentence-BERT (SBERT) embeddings, FAISS for similarity search
- **Database:** SQLite
- **Frontend:** HTML, CSS, JavaScript (Jinja2 templates)

## Project Structure

```
skinsafe_miniproject/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── model/
│   ├── processed_skincare_data.csv # Cleaned product dataset
│   ├── product_embeddings.npy      # Precomputed SBERT embeddings
│   └── skincare_faiss.index        # FAISS similarity search index
├── static/
│   ├── css/style.css
│   └── js/main.js
└── templates/
    ├── base.html
    ├── index.html
    ├── login.html / signup.html / logout.html
    ├── profile.html
    ├── categories.html / category_page.html
    ├── search.html / product.html
    ├── skin_test.html / skin_result.html
    ├── recommend.html
    ├── saved.html
    └── notfound.html
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/hennahameedkp29-eng/Skinsafe-miniproject.git
   cd Skinsafe-miniproject
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables

   Create a `.env` file in the project root:
   ```
   FLASK_SECRET=your_own_secret_key_here
   ```

5. Initialize the database

   > Note: `skincare.db` is not included in this repository, since it contains user account data. You'll need to initialize a fresh database before running the app locally. *(If you have a schema/setup script, mention it here — e.g. `python init_db.py`.)*

6. Run the app
   ```bash
   python app.py
   ```

   Then open `http://localhost:5000` in your browser.

## How the Recommendation Engine Works

1. Product descriptions and attributes are encoded into dense vector embeddings using SBERT.
2. Embeddings are indexed with FAISS for fast approximate nearest-neighbor search.
3. When a user completes the skin quiz or searches for a product, their input is embedded and matched against the product index to return the most semantically relevant results.

## Future Improvements

- Add a database seeding/migration script for easier setup
- Expand the product catalog and category coverage
- Add ingredient-level filtering (e.g. avoid known irritants)
- Deploy a live demo

## Author

Henna Hameed