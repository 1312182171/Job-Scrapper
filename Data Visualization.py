import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.ticker as mtick

def create_visualizations(input_file, model_dir='models', output_dir='visualizations'):
    """Create visualizations for job salary data and model results"""
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(input_file)
    
    # Filter to include only rows with salary data
    salary_df = df.dropna(subset=['salary_avg'])
    print(f"Working with {len(salary_df)} records with salary information")
    
    # Load models
    models = {}
    for model_name in ['random_forest', 'gradient_boosting']:
        model_path = f"{model_dir}/{model_name}_model.pkl"
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)
    
    # Load feature importance data
    importance_data = {}
    for model_name in models:
        importance_path = f"{model_dir}/{model_name}_feature_importance.csv"
        if os.path.exists(importance_path):
            importance_data[model_name] = pd.read_csv(importance_path)
    
    # 1. Salary Distribution by Job Title
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x='standardized_title', y='salary_avg', data=salary_df)
    plt.title('Salary Distribution by Job Title', fontsize=16)
    plt.xlabel('Job Title')
    plt.ylabel('Annual Salary ($)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Format y-axis to show dollar values
    formatter = mtick.StrMethodFormatter('${x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)
    
    plt.savefig(f"{output_dir}/salary_by_job_title.png", dpi=300)
    plt.close()
    
    # 2. Salary Distribution by Location
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x='standardized_location', y='salary_avg', data=salary_df)
    plt.title('Salary Distribution by Location', fontsize=16)
    plt.xlabel('Location')
    plt.ylabel('Annual Salary ($)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Format y-axis to show dollar values
    ax.yaxis.set_major_formatter(formatter)
    
    plt.savefig(f"{output_dir}/salary_by_location.png", dpi=300)
    plt.close()
    
    # 3. Salary vs. Experience
    plt.figure(figsize=(10, 6))
    sns.regplot(x='experience_years', y='salary_avg', data=salary_df, scatter_kws={'alpha':0.5})
    plt.title('Salary vs. Years of Experience', fontsize=16)
    plt.xlabel('Years of Experience')
    plt.ylabel('Annual Salary ($)')
    plt.tight_layout()
    
    # Format y-axis to show dollar values
    plt.gca().yaxis.set_major_formatter(formatter)
    
    plt.savefig(f"{output_dir}/salary_vs_experience.png", dpi=300)
    plt.close()
    
    # 4. Salary vs. Number of Skills
    plt.figure(figsize=(10, 6))
    sns.regplot(x='skills_count', y='salary_avg', data=salary_df, scatter_kws={'alpha':0.5})
    plt.title('Salary vs. Number of Required Skills', fontsize=16)
    plt.xlabel('Number of Skills')
    plt.ylabel('Annual Salary ($)')
    plt.tight_layout()
    
    # Format y-axis to show dollar values
    plt.gca().yaxis.set_major_formatter(formatter)
    
    plt.savefig(f"{output_dir}/salary_vs_skills_count.png", dpi=300)
    plt.close()
    
    # 5. Model Predictions vs. Actual Salaries
    if models:
        # Get predictions from each model
        X = salary_df.drop(columns=['job_title', 'company', 'location', 'salary', 
                                   'description', 'skills', 'salary_low', 'salary_avg'])
        y_actual = salary_df['salary_avg']
        
        plt.figure(figsize=(10, 10))
        
        for i, (model_name, model) in enumerate(models.items()):
            y_pred = model.predict(X)
            
            # Calculate metrics
            mae = mean_absolute_error(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            r2 = r2_score(y_actual, y_pred)
            
            plt.subplot(len(models), 1, i+1)
            plt.scatter(y_actual, y_pred, alpha=0.5)
            plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')
            plt.title(f'{model_name.replace("_", " ").title()} Predictions vs. Actual', fontsize=14)
            plt.xlabel('Actual Salary ($)')
            plt.ylabel('Predicted Salary ($)')
            plt.annotate(f'MAE: ${mae:,.0f}\nRMSE: ${rmse:,.0f}\nR²: {r2:.3f}',
                        xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12)
            
            # Format axes to show dollar values
            plt.gca().xaxis.set_major_formatter(formatter)
            plt.gca().yaxis.set_major_formatter(formatter)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_predictions.png", dpi=300)
        plt.close()
    
    # 6. Feature Importance Visualization
    if importance_data:
        # For each model
        for model_name, importance_df in importance_data.items():
            # Filter to only include top 15 features
            top_features = importance_df.head(15).copy()
            
            # Clean up skill features for display
            top_features['feature_display'] = top_features['feature'].apply(
                lambda x: x.replace('skill_', '').replace('_', ' ').title() 
                if x.startswith('skill_') else 
                x.replace('_', ' ').title()
            )
            
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(x='importance', y='feature_display', data=top_features)
            plt.title(f'Top 15 Features - {model_name.replace("_", " ").title()}', fontsize=16)
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            
            plt.savefig(f"{output_dir}/{model_name}_feature_importance.png", dpi=300)
            plt.close()
            
    # 7. Skill Frequency Heatmap
    skill_columns = [col for col in df.columns if col.startswith('skill_')]
    
    if len(skill_columns) > 0:
        # Select top 20 most common skills
        skill_freq = df[skill_columns].mean().sort_values(ascending=False).head(20)
        
        # Clean skill names
        skill_names = [col.replace('skill_', '').replace('_', ' ').title() for col in skill_freq.index]
        
        plt.figure(figsize=(12, 10))
        ax = sns.barplot(x=skill_freq.values, y=skill_names)
        plt.title('Top 20 Most Frequently Required Skills', fontsize=16)
        plt.xlabel('Frequency (% of Job Postings)')
        plt.ylabel('Skill')
        
        # Format x-axis as percentage
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_skills_frequency.png", dpi=300)
        plt.close()
        
        # 8. Skill Heatmap by Job Field
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
        
        # Create a dataframe to hold skill frequencies by field
        field_skill_data = []
        
        # For each field, calculate the frequency of each top skill
        for field_name, job_titles in fields.items():
            field_df = df[df['standardized_title'].isin(job_titles)]
            
            if len(field_df) == 0:
                continue
                
            # Get the top 10 skills for this field
            field_skill_freq = field_df[skill_columns].mean().sort_values(ascending=False).head(10)
            
            for skill, freq in field_skill_freq.items():
                skill_name = skill.replace('skill_', '').replace('_', ' ').title()
                field_skill_data.append({
                    'Field': field_name,
                    'Skill': skill_name,
                    'Frequency': freq
                })
        
        if field_skill_data:
            skill_freq_df = pd.DataFrame(field_skill_data)
            
            # Create a pivot table for the heatmap
            pivot_df = skill_freq_df.pivot(index='Skill', columns='Field', values='Frequency')
            
            plt.figure(figsize=(10, 12))
            ax = sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.1%', linewidths=0.5)
            plt.title('Top Skills by Job Field', fontsize=16)
            plt.tight_layout()
            
            plt.savefig(f"{output_dir}/skill_heatmap_by_field.png", dpi=300)
            plt.close()
    
    # 9. Salary Distribution by Company Size
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='company_size', y='salary_avg', data=salary_df)
    plt.title('Salary Distribution by Company Size', fontsize=16)
    plt.xlabel('Company Size')
    plt.ylabel('Annual Salary ($)')
    plt.tight_layout()
    
    # Format y-axis to show dollar values
    ax.yaxis.set_major_formatter(formatter)
    
    plt.savefig(f"{output_dir}/salary_by_company_size.png", dpi=300)
    plt.close()
    
    print(f"Visualizations created and saved to {output_dir}")

# Main execution
if __name__ == "__main__":
    create_visualizations('job_data_processed.csv')