import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def predict_salaries(input_file, models_output_dir='models'):
    """Build models to predict salaries based on job features"""
    print("Loading preprocessed data...")
    df = pd.read_csv(input_file)
    
    # Filter to include only rows with salary data
    salary_df = df.dropna(subset=['salary_avg'])
    print(f"Working with {len(salary_df)} records with salary information")
    
    if len(salary_df) < 50:
        print("Warning: Very small dataset for modeling. Results may not be reliable.")
    
    # Define features and target
    y = salary_df['salary_avg']
    
    # Get all skill columns
    skill_columns = [col for col in salary_df.columns if col.startswith('skill_')]
    
    # Define feature columns by category
    categorical_features = ['standardized_title', 'standardized_location', 'company_size']
    numeric_features = ['experience_years', 'skills_count'] 
    
    # All features
    X = salary_df[categorical_features + numeric_features + skill_columns]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessor
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Binary skill features don't need transformation
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # This will pass through all other columns (skills)
    )
    
    # Define models
    models = {
        'random_forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        
        'gradient_boosting': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
    }
    
    # Train and evaluate models
    results = {}
    feature_importance = {}
    
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name} - MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, R²: {r2:.3f}")
        
        results[name] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'model': model
        }
        
        # Save model
        joblib.dump(model, f"{models_output_dir}/{name}_model.pkl")
        
        # Feature importance analysis (if applicable)
        if hasattr(model[-1], 'feature_importances_'):
            # Get feature names from the preprocessor
            feature_names = []
            
            # Get feature names for numeric features (original names)
            feature_names.extend(numeric_features)
            
            # Get one-hot encoded feature names for categorical features
            cat_feature_names = []
            for i, col in enumerate(categorical_features):
                categories = model[0].transformers_[1][1].named_steps['onehot'].categories_[i]
                for category in categories:
                    cat_feature_names.append(f"{col}_{category}")
            feature_names.extend(cat_feature_names)
            
            # Add skill feature names (they were passed through)
            feature_names.extend(skill_columns)
            
            # Check if feature names match the expected length
            if len(feature_names) != len(model[-1].feature_importances_):
                print(f"Warning: Feature names length ({len(feature_names)}) doesn't match importances length ({len(model[-1].feature_importances_)})")
                # Fallback to generic names if lengths don't match
                feature_names = [f"feature_{i}" for i in range(len(model[-1].feature_importances_))]
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model[-1].feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance[name] = importance_df
            
            # Save feature importance
            importance_df.to_csv(f"{models_output_dir}/{name}_feature_importance.csv", index=False)
            
            print(f"\nTop 10 most important features for {name}:")
            print(importance_df.head(10))
    
    return results, feature_importance

def analyze_skill_importance(input_file, feature_importance, output_file, field_filter=None):
    """Analyze skill importance by job field"""
    print("\nAnalyzing skill importance by job field...")
    df = pd.read_csv(input_file)
    
    # Use random forest importance data
    if 'random_forest' in feature_importance:
        importance_df = feature_importance['random_forest']
    else:
        importance_df = list(feature_importance.values())[0]  # Take whatever is available
    
    # Filter for only skill features
    skill_importance = importance_df[importance_df['feature'].str.startswith('skill_')].copy()
    
    # Clean up skill names
    skill_importance['skill'] = skill_importance['feature'].str.replace('skill_', '').str.replace('_', ' ').str.title()
    
    # Create skill importance dictionary
    skill_importance_dict = {}
    
    # Define job fields
    fields = {
        'Software Engineering': ['Software Engineer', 'Senior Software Engineer', 'Junior Software Engineer', 
                                 'Principal Software Engineer', 'Frontend Engineer', 'Backend Engineer', 
                                 'Full Stack Engineer'],
        'Data Science': ['Data Scientist', 'Senior Data Scientist', 'Junior Data Scientist', 
                         'Principal Data Scientist', 'Data Analyst'],
        'Machine Learning': ['ML Engineer', 'Senior ML Engineer', 'Junior ML Engineer', 
                             'Principal ML Engineer', 'ML Research Scientist']
    }
    
    # Filter data if field_filter is provided
    if field_filter:
        fields = {k: v for k, v in fields.items() if k == field_filter}
    
    # For each job field, get the top skills
    field_skill_importance = {}
    
    for field_name, job_titles in fields.items():
        # Filter the data to only include jobs in this field
        field_df = df[df['standardized_title'].isin(job_titles)]
        
        if len(field_df) == 0:
            print(f"No data for {field_name}")
            continue
            
        print(f"\nAnalyzing {len(field_df)} jobs in {field_name}")
        
        # Get all skill columns
        skill_columns = [col for col in field_df.columns if col.startswith('skill_')]
        
        # Calculate the frequency of each skill in this field
        skill_freq = {}
        for col in skill_columns:
            skill_name = col.replace('skill_', '').replace('_', ' ').title()
            freq = field_df[col].mean()  # Proportion of jobs requiring this skill
            skill_freq[skill_name] = freq
        
        # Sort skills by frequency
        sorted_skills = sorted(skill_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Get the top 20 skills by frequency
        top_skills_by_freq = [item[0] for item in sorted_skills[:20]]
        
        # Get the importance scores for these skills
        top_skills_importance = {}
        for skill in top_skills_by_freq:
            skill_key = 'skill_' + skill.lower().replace(' ', '_')
            # Find the importance from the importance dataframe
            importance_row = skill_importance[skill_importance['feature'] == skill_key]
            if not importance_row.empty:
                top_skills_importance[skill] = float(importance_row['importance'].iloc[0])
            else:
                top_skills_importance[skill] = 0.0
        
        # Sort by importance
        sorted_by_importance = sorted(top_skills_importance.items(), key=lambda x: x[1], reverse=True)
        
        field_skill_importance[field_name] = {
            'top_skills_by_freq': top_skills_by_freq,
            'top_skills_by_importance': [item[0] for item in sorted_by_importance],
            'skill_freq': skill_freq,
            'skill_importance': top_skills_importance
        }
    
    # Save results
    with open(output_file, 'w') as f:
        f.write("# Skill Importance Analysis by Job Field\n\n")
        
        for field, data in field_skill_importance.items():
            f.write(f"## {field}\n\n")
            
            f.write("### Most Common Skills\n")
            for i, skill in enumerate(data['top_skills_by_freq'][:10], 1):
                freq = data['skill_freq'][skill] * 100
                f.write(f"{i}. {skill} - Present in {freq:.1f}% of job postings\n")
            
            f.write("\n### Most Important Skills for Salary\n")
            for i, skill in enumerate(data['top_skills_by_importance'][:10], 1):
                importance = data['skill_importance'][skill]
                f.write(f"{i}. {skill} - Importance score: {importance:.4f}\n")
            
            f.write("\n")
    
    print(f"Skill importance analysis saved to {output_file}")
    return field_skill_importance

# Main execution
if __name__ == "__main__":
    import os
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train salary prediction models
    results, importance = predict_salaries('job_data_processed.csv', 'models')
    
    # Analyze skill importance by field
    field_skill_importance = analyze_skill_importance(
        'job_data_processed.csv', 
        importance, 
        'skill_importance_analysis.md'
    )