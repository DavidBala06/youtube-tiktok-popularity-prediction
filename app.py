import streamlit as st
import pandas as pd
import numpy as np
import joblib
from cleaning import preprocess_features

st.set_page_config(page_title="Popularity Predictor", page_icon="🎬", layout="wide")
st.title("YouTube/TikTok Popularity Predictor")

# Load model and feature columns
try:
    regressor = joblib.load("xgb_regressor.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    model_ready = True
except Exception as e:
    st.error(
        "Model not trained yet. Please run `python main.py` first to create `xgb_regressor.pkl` and `feature_columns.pkl` files.")
    st.stop()


def analyze_improvements_advanced(original_features, original_prediction, feature_names):
    """
    Advanced analysis with multiple improvement suggestions
    """
    suggestions = []

    # Comprehensive optimization rules with multiple test values
    optimization_rules = {
        'title_len_chars': {
            'test_values': [25, 35, 45, 55, 65],
            'message': "Test different title lengths",
            'type': 'title'
        },
        'title_exclamations': {
            'test_values': [0, 1, 2, 3],
            'message': "Adjust exclamation marks",
            'type': 'title'
        },
        'title_questions': {
            'test_values': [0, 1, 2],
            'message': "Try question format",
            'type': 'title'
        },
        'title_emojis': {
            'test_values': [0, 1, 2, 3],
            'message': "Add or remove emojis",
            'type': 'title'
        },
        'title_upper_words': {
            'test_values': [0, 1, 2, 3, 4],
            'message': "Adjust uppercase words",
            'type': 'title'
        },
        'title_has_numbers': {
            'test_values': [0, 1],
            'message': "Include numbers in title",
            'type': 'title'
        },
        'is_weekend': {
            'test_values': [0, 1],
            'message': "Try weekend publishing",
            'type': 'scheduling'
        },
        'month': {
            'test_values': [1, 3, 6, 9, 12],
            'message': "Test different months",
            'type': 'scheduling'
        },
        'dayofweek': {
            'test_values': [0, 2, 4, 6],
            'message': "Try different weekdays",
            'type': 'scheduling'
        }
    }

    # Test each feature with multiple values
    for feat, rules in optimization_rules.items():
        if feat in feature_names:
            idx = feature_names.index(feat)
            current_val = original_features[idx]

            for test_val in rules['test_values']:
                if test_val != current_val:
                    # Test the impact
                    test_features = original_features.copy()
                    test_features[idx] = test_val
                    test_pred = regressor.predict([test_features])[0]

                    improvement_pct = ((test_pred - original_prediction) / original_prediction) * 100

                    # Lower threshold to 0.5% improvement
                    if improvement_pct > 0.5:
                        suggestion_text = f"{rules['message']}"
                        if feat != 'is_weekend':
                            suggestion_text += f" (change {current_val} → {test_val})"

                        suggestions.append({
                            'suggestion': suggestion_text,
                            'improvement': f"+{improvement_pct:.1f}%",
                            'new_views': f"{test_pred:,.0f}",
                            'current_views': f"{original_prediction:,.0f}",
                            'feature': feat,
                            'type': rules['type'],
                            'impact': improvement_pct
                        })

    # Also test some combined improvements
    combined_suggestions = test_combined_improvements(original_features, original_prediction, feature_names)
    suggestions.extend(combined_suggestions)

    # Sort by highest positive impact
    suggestions.sort(key=lambda x: x['impact'], reverse=True)

    return suggestions[:15]


def test_combined_improvements(original_features, original_prediction, feature_names):
    """Test some common combination improvements"""
    suggestions = []
    common_combinations = [
        ('title_has_numbers', 1, 'title_exclamations', 1, "Add numbers and exclamation"),
        ('title_emojis', 1, 'title_questions', 1, "Combine emoji with question"),
        ('is_weekend', 1, 'title_exclamations', 1, "Weekend post with excitement"),
        ('title_has_numbers', 1, 'title_emojis', 1, "Numbers with emojis"),
        ('title_questions', 1, 'is_weekend', 1, "Question format on weekend")
    ]

    for combo in common_combinations:
        feat1, val1, feat2, val2, message = combo

        if feat1 in feature_names and feat2 in feature_names:
            test_features = original_features.copy()
            idx1 = feature_names.index(feat1)
            idx2 = feature_names.index(feat2)

            if test_features[idx1] != val1 or test_features[idx2] != val2:
                test_features[idx1] = val1
                test_features[idx2] = val2

                test_pred = regressor.predict([test_features])[0]
                improvement_pct = ((test_pred - original_prediction) / original_prediction) * 100

                if improvement_pct > 1:
                    suggestions.append({
                        'suggestion': f"{message}",
                        'improvement': f"+{improvement_pct:.1f}%",
                        'new_views': f"{test_pred:,.0f}",
                        'current_views': f"{original_prediction:,.0f}",
                        'feature': 'combined',
                        'type': 'combination',
                        'impact': improvement_pct
                    })

    return suggestions


def display_suggestions_by_category(suggestions):
    """Group and display suggestions by category - only positive ones"""
    # Filter only suggestions with positive impact
    positive_suggestions = [s for s in suggestions if s['improvement'].startswith('+')]

    if not positive_suggestions:
        st.info("📉 No suggestions found that would improve popularity.")
        return

    categories = {}
    for suggestion in positive_suggestions:
        category = suggestion['type']
        if category not in categories:
            categories[category] = []
        categories[category].append(suggestion)

    for category, category_suggestions in categories.items():
        st.subheader(f"🎯 {category.capitalize()}")

        for i, suggestion in enumerate(category_suggestions, 1):
            improvement_val = float(suggestion['improvement'].replace('%', '').replace('+', ''))

            with st.expander(f"Suggestion #{i}: {suggestion['suggestion']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Estimated Improvement", suggestion['improvement'])
                with col2:
                    st.metric("Potential Views", suggestion['new_views'])

                # Normalized progress bar
                progress_value = min(improvement_val / 50, 1.0)
                st.progress(progress_value)

                st.caption(f"Compared to {suggestion['current_views']} current estimated views")


# Main interface
uploaded = st.file_uploader("Upload a CSV file (e.g., videos-stats.csv or similar)", type=["csv"])

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    st.subheader("Uploaded Data (first 5 rows)")
    st.dataframe(raw.head())

    # Preprocessing FOR PREDICTION
    X_new = preprocess_features(raw, is_training=False)
    X_new = X_new.reindex(columns=feature_cols, fill_value=0.0).astype("float32")
    X_new.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_new.fillna(0.0, inplace=True)

    # Debug: Show processed features
    st.subheader("🔍 Debug: Processed Features")
    st.write(f"Number of features: {len(X_new.columns)}")
    st.write(f"Expected features: {len(feature_cols)}")

    # Check which expected features are missing
    missing_features = [feat for feat in feature_cols if feat not in X_new.columns]
    if missing_features:
        st.warning(f"Missing features: {missing_features}")

    # Predict
    preds = regressor.predict(X_new).astype(float)

    out = raw.copy()
    out["Predicted_Views"] = preds

    st.subheader("🔮 Predictions")
    st.dataframe(out.head(20))

    # What-if analysis for the first video
    st.subheader("💡 Improvement Suggestions for the First Video")

    if len(X_new) > 0:
        first_video_features = X_new.iloc[0].values
        first_video_pred = preds[0]

        # Show current feature values for debugging
        st.subheader("🔍 Current Feature Values")
        important_features = ['title_len_chars', 'title_exclamations', 'title_questions',
                              'title_emojis', 'title_upper_words', 'title_has_numbers',
                              'is_weekend', 'month', 'dayofweek']

        feature_values = {}
        for feat in important_features:
            if feat in feature_cols:
                idx = feature_cols.index(feat)
                feature_values[feat] = first_video_features[idx]

        st.json(feature_values)

        suggestions = analyze_improvements_advanced(
            first_video_features,
            first_video_pred,
            feature_cols
        )

        st.write(f"**Found {len(suggestions)} suggestions**")

        if suggestions:
            display_suggestions_by_category(suggestions)

            # Show total potential improvement
            positive_suggestions = [s for s in suggestions if s['improvement'].startswith('+')]
            if positive_suggestions:
                total_improvement = sum(float(s['improvement'].replace('%', '').replace('+', ''))
                                        for s in positive_suggestions)
                st.success(f"**Total potential improvement: +{total_improvement:.1f}%**")
        else:
            st.warning("No improvements found even with aggressive testing.")
            st.info(
                "This could mean: 1) Your video is already optimal, 2) The model needs more training data, or 3) Try more extreme value changes")

    # Interactive testing tool
    st.subheader("🧪 Manual Testing Tool")

    if len(X_new) > 0:
        col1, col2, col3 = st.columns(3)

        with col1:
            current_title_len = int(X_new.iloc[0, feature_cols.index('title_len_chars')])
            new_title_len = st.slider("Title Length", 10, 100, current_title_len)

        with col2:
            current_excl = int(X_new.iloc[0, feature_cols.index('title_exclamations')])
            new_exclamations = st.slider("Exclamation Marks", 0, 3, current_excl)

        with col3:
            current_quest = int(X_new.iloc[0, feature_cols.index('title_questions')])
            new_questions = st.slider("Question Marks", 0, 2, current_quest)

        if st.button("📊 Calculate Impact of Changes"):
            test_features = X_new.iloc[0].copy()
            test_features[feature_cols.index('title_len_chars')] = new_title_len
            test_features[feature_cols.index('title_exclamations')] = new_exclamations
            test_features[feature_cols.index('title_questions')] = new_questions

            new_pred = regressor.predict([test_features])[0]
            improvement = ((new_pred - first_video_pred) / first_video_pred) * 100

            st.metric("Estimated Views", f"{new_pred:,.0f}",
                      f"{improvement:+.1f}%", delta_color="normal")

            st.info(f"**Changes tested:** Title Length: {new_title_len} chars, "
                    f"Exclamations: {new_exclamations}, Questions: {new_questions}")

    # Download results
    st.download_button(
        label="📥 Download All Predictions (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv"
    )

else:
    st.info("📤 Upload a `.csv` file with video data to see predictions and suggestions.")

# Sample CSV structure help
with st.expander("📋 Sample CSV Structure"):
    st.write("""
    Your CSV should contain these columns:
    - **Title**: Video title text
    - **Published At**: Date in YYYY-MM-DD format
    - **Keyword**: Category or keyword (optional)

    Example:
    ```
    Title,Published At,Keyword
    My Amazing Video,2024-01-15,vlog
    Top 10 Tips,2024-01-16,tutorial
    How to become popular,2024-01-17,advice
    ```
    """)