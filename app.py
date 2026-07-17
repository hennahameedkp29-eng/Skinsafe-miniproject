"""
Skincare Recommender App - Fixed with Index-Based Product IDs
"""

import os
import ast
import pickle
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from sentence_transformers import SentenceTransformer
import faiss

# ============================================================================
# APP CONFIGURATION
# ============================================================================

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "supersecretkey_change_in_production")

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "skincare.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# ============================================================================
# DATABASE MODELS
# ============================================================================

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password = db.Column(db.String(255), nullable=False)

class UserProfile(db.Model):
    __tablename__ = 'user_profiles'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)
    skin_type = db.Column(db.String(100))
    preferences = db.Column(db.Text)
    targets = db.Column(db.Text)

class SavedProduct(db.Model):
    __tablename__ = 'saved_products'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    product_index = db.Column(db.Integer, nullable=False)  # Changed to product_index
    __table_args__ = (db.UniqueConstraint('user_id', 'product_index'),)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ============================================================================
# LOAD DATA & MODELS
# ============================================================================

def parse_list(x):
    """Parse list-like strings from CSV"""
    if pd.isna(x):
        return []
    s = str(x).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [str(i).strip() for i in v]
    except:
        pass
    return [i.strip() for i in s.split(",") if i.strip()]

def load_products():
    """Load product data - RESET INDEX to ensure clean numeric indices"""
    csv_path = os.path.join(basedir, "model", "processed_skincare_data.csv")
    
    print("\n" + "="*70)
    print("📦 LOADING PRODUCTS")
    print("="*70)
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        
        # CRITICAL: Reset index to ensure clean 0-based numeric indices
        df = df.reset_index(drop=True)
        
        print(f"✓ Loaded {len(df)} rows with indices 0-{len(df)-1}")
        
        # Create title column
        if "product title" in df.columns:
            df["title"] = df["product title"].fillna("")
        elif "product_name" in df.columns:
            df["title"] = df["product_name"].fillna("")
        else:
            brand = df.get("brand", "").fillna("")
            name = df.get("product_name", df.get("product title", "")).fillna("")
            df["title"] = (brand + " " + name).astype(str)
        
        # Parse list columns
        for col in ["suited_skin_types", "concerns", "benefits_clean"]:
            if col in df.columns:
                df[col] = df[col].apply(parse_list)
            else:
                df[col] = [[] for _ in range(len(df))]
        
        # Standard columns
        df["image_url"] = df.get("image_url_y", df.get("image_url", "")).fillna("")
        df["category"] = df.get("category", "Unknown").fillna("Unknown").astype(str)
        df["brand"] = df.get("brand", "").fillna("").astype(str)
        
        print(f"✅ Products ready: {len(df)}")
        print("="*70 + "\n")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def load_faiss_system():
    """Load FAISS index and embeddings"""
    try:
        model_dir = os.path.join(basedir, "model")
        
        print("\n" + "="*70)
        print("🤖 LOADING FAISS RECOMMENDATION SYSTEM")
        print("="*70)
        
        # Load SBERT model
        print("Loading SBERT model...")
        sbert = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load embeddings
        emb_path = os.path.join(model_dir, "product_embeddings.npy")
        print(f"Loading embeddings from {emb_path}")
        embeddings = np.load(emb_path).astype('float32')
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        
        # Load FAISS index
        index_path = os.path.join(model_dir, "skincare_faiss.index")
        print(f"Loading FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        
        print(f"✅ Loaded {len(embeddings)} embeddings")
        print("="*70 + "\n")
        
        return sbert, embeddings, index
        
    except Exception as e:
        print(f"❌ FAISS loading failed: {e}")
        return None, None, None

def load_skin_model():
    """Load skin type prediction model"""
    try:
        model_dir = os.path.join(basedir, "model")
        
        print("\n" + "="*70)
        print("🧪 LOADING SKIN TYPE MODEL")
        print("="*70)
        
        model = joblib.load(os.path.join(model_dir, "skin_type_model.pkl"))
        encoders = joblib.load(os.path.join(model_dir, "label_encoders.pkl"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        
        features = ['Age', 'Gender', 'Hydration_Level', 'Oil_Level', 'Sensitivity', 'Temperature', 'Humidity']
        
        print(f"✓ Model: {type(model).__name__}")
        print(f"✓ Features: {features}")
        print("✅ Skin model ready")
        print("="*70 + "\n")
        
        return model, scaler, encoders, features
        
    except Exception as e:
        print(f"❌ Skin model loading failed: {e}")
        return None, None, None, None

# Load everything
df_products = load_products()
sbert_model, product_embeddings, faiss_index = load_faiss_system()
skin_model, scaler, label_encoders, feature_names = load_skin_model()

# ============================================================================
# HELPER FUNCTIONS - NOW INDEX-BASED
# ============================================================================

def search_products(query):
    """Search products using FAISS - returns indices"""
    if df_products.empty or sbert_model is None or faiss_index is None:
        return []
    
    try:
        # Encode query
        query_emb = sbert_model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_emb)
        
        # Search
        k = min(50, len(df_products))
        distances, indices = faiss_index.search(query_emb, k)
        
        # Return valid indices
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(df_products):
                results.append(int(idx))
        
        return results
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        # Fallback to simple text search - return indices
        q = query.lower()
        mask = df_products['title'].str.lower().str.contains(q, na=False, regex=False)
        return df_products[mask].head(50).index.tolist()

def find_alternatives(product_idx, n=6):
    """Find similar products using FAISS - returns indices"""
    if product_embeddings is None or faiss_index is None or df_products.empty:
        return []
    
    try:
        idx = int(product_idx)
        if idx < 0 or idx >= len(product_embeddings):
            return []
        
        # Get embedding
        emb = product_embeddings[idx].reshape(1, -1)
        
        # Search for similar
        distances, indices = faiss_index.search(emb, n + 1)
        
        # Exclude self and return valid indices
        results = []
        for i in indices[0]:
            if i != idx and 0 <= i < len(df_products):
                results.append(int(i))
                if len(results) >= n:
                    break
        
        return results
        
    except Exception as e:
        print(f"❌ Alternatives error: {e}")
        return []

def get_product_by_index(idx):
    """Get product by index - returns dict with 'index' key"""
    try:
        idx = int(idx)
        if idx < 0 or idx >= len(df_products):
            return None
        
        prod = df_products.iloc[idx].to_dict()
        prod['index'] = idx  # Add index to product dict
        return prod
        
    except Exception as e:
        print(f"❌ Error getting product {idx}: {e}")
        return None

def calculate_match(profile, product):
    """Calculate match score"""
    if not profile:
        return {'overall': 0, 'skin': 0, 'prefs': 0, 'targets': 0}
    
    skin_score = 0
    pref_score = 0
    target_score = 0
    
    # Skin type (40%)
    if profile.skin_type:
        user_skin = profile.skin_type.lower().strip()
        prod_skins = product.get('suited_skin_types', [])
        if isinstance(prod_skins, str):
            prod_skins = parse_list(prod_skins)
        prod_skins_str = ' '.join([str(s).lower() for s in prod_skins])
        if user_skin in prod_skins_str:
            skin_score = 40
    
    # Preferences (40%)
    if profile.preferences:
        user_prefs = [p.lower().strip() for p in profile.preferences.split(',') if p.strip()]
        prod_concerns = product.get('concerns', [])
        if isinstance(prod_concerns, str):
            prod_concerns = parse_list(prod_concerns)
        prod_concerns_str = ' '.join([str(c).lower() for c in prod_concerns])
        
        if user_prefs:
            matches = sum(1 for p in user_prefs if p in prod_concerns_str)
            pref_score = (matches / len(user_prefs)) * 40
    
    # Targets (20%)
    if profile.targets:
        user_targets = [t.lower().strip() for t in profile.targets.split(',') if t.strip()]
        prod_benefits = product.get('benefits_clean', [])
        if isinstance(prod_benefits, str):
            prod_benefits = parse_list(prod_benefits)
        prod_benefits_str = ' '.join([str(b).lower() for b in prod_benefits])
        
        if user_targets:
            matches = sum(1 for t in user_targets if t in prod_benefits_str)
            target_score = (matches / len(user_targets)) * 20
    
    overall = round(skin_score + pref_score + target_score)
    return {'overall': overall, 'skin': round(skin_score), 'prefs': round(pref_score), 'targets': round(target_score)}

def format_product(prod):
    """Format product for display - ensures all fields are safe"""
    if prod is None:
        return None
    
    prod = prod.copy()
    
    # Ensure index exists
    if 'index' not in prod:
        print("⚠️  Product missing index!")
        prod['index'] = 0
    
    # Ensure title exists
    if 'title' not in prod or pd.isna(prod.get('title')) or str(prod.get('title')).strip() == '':
        if 'product_name' in prod and not pd.isna(prod.get('product_name')):
            prod['title'] = str(prod['product_name'])
        else:
            prod['title'] = 'Unknown Product'
    
    prod['product_name'] = prod['title']
    
    # Convert lists to strings for display
    for key in ['suited_skin_types', 'concerns', 'benefits_clean']:
        val = prod.get(key)
        if isinstance(val, list):
            prod[key] = ', '.join([str(s) for s in val if s])
        elif pd.isna(val) or val is None or val == '':
            prod[key] = ''
        elif not isinstance(val, str):
            prod[key] = str(val)
    
    # Ensure brand exists
    if 'brand' not in prod or pd.isna(prod.get('brand')) or str(prod.get('brand')).strip() == '':
        prod['brand'] = 'Unknown Brand'
    else:
        prod['brand'] = str(prod['brand'])
    
    # Ensure category exists
    if 'category' not in prod or pd.isna(prod.get('category')):
        prod['category'] = 'Unknown'
    else:
        prod['category'] = str(prod['category'])
    
    # Ensure image_url exists
    if 'image_url' not in prod or pd.isna(prod.get('image_url')) or str(prod.get('image_url')).strip() == '':
        prod['image_url'] = ''
    else:
        prod['image_url'] = str(prod['image_url'])
    
    return prod

def is_saved(idx):
    """Check if product is saved by index"""
    if not current_user.is_authenticated:
        return False
    try:
        idx = int(idx)
        return SavedProduct.query.filter_by(user_id=current_user.id, product_index=idx).first() is not None
    except:
        return False

# ============================================================================
# ROUTES - AUTHENTICATION
# ============================================================================

@app.route('/')
def home():
    username = current_user.email.split('@')[0] if current_user.is_authenticated else None
    return render_template('index.html', username=username)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        
        if not email or not password:
            flash('Email and password required', 'danger')
            return redirect(url_for('signup'))
        
        if password != confirm:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('signup'))
        
        if len(password) < 6:
            flash('Password must be 6+ characters', 'danger')
            return redirect(url_for('signup'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'warning')
            return redirect(url_for('login'))
        
        hashed = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(email=email, password=hashed)
        
        try:
            db.session.add(user)
            db.session.commit()
            flash('Account created! Please login.', 'success')
            return redirect(url_for('login'))
        except:
            db.session.rollback()
            flash('Error creating account', 'danger')
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()
        
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash(f'Welcome back!', 'success')
            return redirect(request.args.get('next') or url_for('home'))
        
        flash('Invalid credentials', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out', 'info')
    return redirect(url_for('home'))

# ============================================================================
# ROUTES - PRODUCTS & SEARCH (INDEX-BASED)
# ============================================================================

@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    
    if not query:
        return render_template('search.html', results=[], query='')
    
    if df_products.empty:
        flash('Database not available', 'warning')
        return render_template('search.html', results=[], query=query)
    
    # Get indices from search
    indices = search_products(query)
    
    if not indices:
        return render_template('notfound.html', query=query)
    
    # Get products by indices
    products = []
    profile = UserProfile.query.filter_by(user_id=current_user.id).first() if current_user.is_authenticated else None
    
    for idx in indices:
        prod = get_product_by_index(idx)
        if prod:
            prod = format_product(prod)
            if profile:
                prod['match_score'] = calculate_match(profile, prod)['overall']
            else:
                prod['match_score'] = None
            products.append(prod)
    
    return render_template('search.html', results=products, query=query)

@app.route('/product/<int:idx>')
def product(idx):
    if df_products.empty:
        flash('Database not available', 'warning')
        return redirect(url_for('home'))
    
    prod = get_product_by_index(idx)
    
    if prod is None:
        return render_template('notfound.html', query=f'Product {idx}')
    
    prod = format_product(prod)
    
    score = None
    if current_user.is_authenticated:
        profile = UserProfile.query.filter_by(user_id=current_user.id).first()
        score = calculate_match(profile, prod)
    
    saved = is_saved(idx)
    
    return render_template('product.html', product=prod, score=score, saved=saved)

@app.route('/product/<int:idx>/alternatives')
def product_alternatives(idx):
    if df_products.empty:
        flash('Database not available', 'warning')
        return redirect(url_for('home'))
    
    base = get_product_by_index(idx)
    
    if base is None:
        return render_template('notfound.html', query=f'Product {idx}')
    
    base = format_product(base)
    
    # Get alternative indices
    alt_indices = find_alternatives(idx, n=6)
    
    products = []
    profile = UserProfile.query.filter_by(user_id=current_user.id).first() if current_user.is_authenticated else None
    
    for alt_idx in alt_indices:
        prod = get_product_by_index(alt_idx)
        if prod:
            prod = format_product(prod)
            if profile:
                prod['match_score'] = calculate_match(profile, prod)['overall']
            else:
                prod['match_score'] = None
            products.append(prod)
    
    return render_template('recommend.html', base_product=base, products=products)

@app.route('/save/<int:idx>')
@login_required
def save_product(idx):
    """Save/unsave product by index"""
    existing = SavedProduct.query.filter_by(user_id=current_user.id, product_index=idx).first()
    
    try:
        if existing:
            db.session.delete(existing)
            db.session.commit()
            flash('Removed from saved', 'info')
        else:
            new_save = SavedProduct(user_id=current_user.id, product_index=idx)
            db.session.add(new_save)
            db.session.commit()
            flash('Product saved!', 'success')
    except Exception as e:
        db.session.rollback()
        print(f"❌ Save error: {e}")
        flash('Error saving', 'danger')
    
    return redirect(request.referrer or url_for('home'))

@app.route('/saved')
@login_required
def saved():
    records = SavedProduct.query.filter_by(user_id=current_user.id).all()
    indices = [r.product_index for r in records]
    
    if not indices or df_products.empty:
        return render_template('saved.html', products=[])
    
    products = []
    profile = UserProfile.query.filter_by(user_id=current_user.id).first()
    
    for idx in indices:
        prod = get_product_by_index(idx)
        if prod:
            prod = format_product(prod)
            prod['match_score'] = calculate_match(profile, prod)['overall']
            products.append(prod)
    
    return render_template('saved.html', products=products)

# ============================================================================
# ROUTES - CATEGORIES (INDEX-BASED)
# ============================================================================

@app.route('/categories')
def categories():
    if df_products.empty:
        return render_template('categories.html', categories=[])
    
    cats = df_products['category'].dropna().unique().tolist()
    cats = sorted([c.strip() for c in cats if c.strip() and c != 'Unknown'])
    
    return render_template('categories.html', categories=cats)

@app.route('/category/<name>')
def category_page(name):
    if df_products.empty:
        flash('Database not available', 'warning')
        return redirect(url_for('home'))
    
    try:
        # Filter by category
        cat_prods = df_products[df_products['category'].str.strip().str.lower() == name.strip().lower()]
        
        if cat_prods.empty:
            flash(f'No products found in {name} category', 'info')
            return render_template('notfound.html', query=name)
        
        products = []
        profile = UserProfile.query.filter_by(user_id=current_user.id).first() if current_user.is_authenticated else None
        
        for idx in cat_prods.index:
            prod = get_product_by_index(idx)
            if prod:
                prod = format_product(prod)
                
                if profile:
                    prod['match_score'] = calculate_match(profile, prod)['overall']
                else:
                    prod['match_score'] = None
                
                products.append(prod)
        
        if not products:
            flash(f'Could not load products from {name}', 'warning')
            return render_template('notfound.html', query=name)
        
        return render_template('category_page.html', category=name, products=products)
        
    except Exception as e:
        print(f"❌ Category page error: {e}")
        import traceback
        traceback.print_exc()
        flash('Error loading category', 'danger')
        return redirect(url_for('categories'))

# ============================================================================
# ROUTES - PROFILE
# ============================================================================

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    prof = UserProfile.query.filter_by(user_id=current_user.id).first()
    
    if request.method == 'POST':
        skin = request.form.get('skin_type', '').strip()
        prefs = ','.join(request.form.getlist('preferences'))
        tgts = ','.join(request.form.getlist('targets'))
        
        try:
            if prof:
                prof.skin_type = skin
                prof.preferences = prefs
                prof.targets = tgts
            else:
                prof = UserProfile(user_id=current_user.id, skin_type=skin, preferences=prefs, targets=tgts)
                db.session.add(prof)
            
            db.session.commit()
            flash('Profile updated!', 'success')
            return redirect(url_for('profile'))
        except:
            db.session.rollback()
            flash('Error updating', 'danger')
    
    return render_template('profile.html', profile=prof)

# ============================================================================
# ROUTES - SKIN TEST
# ============================================================================

@app.route('/skintest')
@login_required
def skintest():
    if skin_model is None:
        flash('Skin test unavailable', 'warning')
        return redirect(url_for('profile'))
    return render_template('skin_test.html')

@app.route('/skintest/submit', methods=['POST'])
@login_required
def skintest_submit():
    if skin_model is None or feature_names is None:
        flash('Skin test unavailable', 'danger')
        return redirect(url_for('profile'))
    
    try:
        age = float(request.form.get('age', 25))
        gender = request.form.get('gender', 'Female').strip()
        hydration = request.form.get('hydration', 'Medium').strip()
        oil = request.form.get('oil', 'Medium').strip()
        sensitivity = request.form.get('sensitivity', 'Medium').strip()
        temperature = float(request.form.get('temperature', 25))
        humidity = float(request.form.get('humidity', 50))
        
        if age <= 0 or age > 120:
            flash('Invalid age', 'danger')
            return redirect(url_for('skintest'))
        
        if not all([gender, hydration, oil, sensitivity]):
            flash('Please fill all fields', 'danger')
            return redirect(url_for('skintest'))
        
        data = {
            'Age': age,
            'Gender': gender,
            'Hydration_Level': hydration,
            'Oil_Level': oil,
            'Sensitivity': sensitivity,
            'Temperature': temperature,
            'Humidity': humidity
        }
        
        X = []
        for feat in feature_names:
            val = data[feat]
            
            if feat in label_encoders:
                enc = label_encoders[feat]
                if val not in enc.classes_:
                    val = enc.classes_[0]
                X.append(enc.transform([val])[0])
            else:
                X.append(val)
        
        X = np.array([X])
        X_scaled = X.copy()
        
        num_indices = [i for i, f in enumerate(feature_names) if f in ['Age', 'Temperature', 'Humidity']]
        X_scaled[:, num_indices] = scaler.transform(X[:, num_indices])
        
        pred = skin_model.predict(X_scaled)[0]
        probs = skin_model.predict_proba(X_scaled)[0] if hasattr(skin_model, 'predict_proba') else None
        
        skin_type = label_encoders['Skin_Type'].inverse_transform([pred])[0]
        
        probs_dict = {}
        if probs is not None:
            for i, cls in enumerate(label_encoders['Skin_Type'].classes_):
                probs_dict[cls] = f'{probs[i]*100:.1f}%'
        
        prof = UserProfile.query.filter_by(user_id=current_user.id).first()
        if prof:
            prof.skin_type = skin_type
        else:
            prof = UserProfile(user_id=current_user.id, skin_type=skin_type, preferences='', targets='')
            db.session.add(prof)
        db.session.commit()
        
        return render_template('skin_result.html', skin_type=skin_type, probs=probs_dict)
        
    except Exception as e:
        print(f'❌ Skin test error: {e}')
        import traceback
        traceback.print_exc()
        flash('Error during prediction', 'danger')
        return redirect(url_for('skintest'))

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return render_template('notfound.html', query='Page'), 404

@app.errorhandler(500)
def server_error(e):
    db.session.rollback()
    return redirect(url_for('home'))

# ============================================================================
# INITIALIZE & RUN
# ============================================================================

def init_db():
    with app.app_context():
        db.create_all()
        print('✅ Database initialized\n')

if __name__ == '__main__':
    init_db()
    
    print('\n' + '='*70)
    print('🚀 SKINCARE APP - STATUS')
    print('='*70)
    print(f'Products: {len(df_products)}')
    print(f'FAISS: {"✓ Ready" if faiss_index is not None else "✗ Not loaded"}')
    print(f'Skin Model: {"✓ Ready" if skin_model is not None else "✗ Not loaded"}')
    print('='*70 + '\n')
    
    app.run(debug=True, host='0.0.0.0', port=5000)
