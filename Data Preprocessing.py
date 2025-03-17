import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import Counter

def clean_salary(salary_str):
    """Extract numeric salary values from salary strings"""
    if not salary_str or pd.isna(salary_str):
        return None, None
    
    # Convert to string if not already
    if not isinstance(salary_str, str):
        salary_str = str(salary_str)
    
    # Extract numbers using regex
    numbers = re.findall(r'[\d,.]+', salary_str)
    
    if not numbers:
        return None, None
    
    # Check if it's a range
    if len(numbers) >= 2:
        # Clean the numbers and convert to float
        try:
            low = float(numbers[0].replace(',', ''))
            high = float(numbers[1].replace(',', ''))
            
            # Check if values are hourly
            is_hourly = 'hour' in salary_str.lower() or '/hr' in salary_str.lower() or 'hourly' in salary_str.lower()
            
            # Convert hourly to annual (assuming 40 hrs/week, 52 weeks/year)
            if is_hourly:
                low = low * 40 * 52
                high = high * 40 * 52
                
            # Make sure low <= high
            if low > high:
                low, high = high, low
                
            # Calculate average
            avg = (low + high) / 2
            return low, avg
        except:
            return None, None
    else:
        # Single number
        try:
            value = float(numbers[0].replace(',', ''))
            
            # Check if the value is hourly
            is_hourly = 'hour' in salary_str.lower() or '/hr' in salary_str.lower() or 'hourly' in salary_str.lower()
            
            # Convert hourly to annual (assuming 40 hrs/week, 52 weeks/year)
            if is_hourly:
                value = value * 40 * 52
                
            return value, value
        except:
            return None, None

def standardize_job_title(title):
    """Standardize job titles to reduce variability"""
    if not title or pd.isna(title):
        return "Unknown"
    
    title = title.lower()
    
    # Software Engineering roles
    if re.search(r'software\s+engineer|developer|programmer|swe\b|front\s*end|back\s*end|full\s*stack', title):
        if 'senior' in title or 'sr' in title.split() or 'lead' in title:
            return "Senior Software Engineer"
        elif 'junior' in title or 'jr' in title.split():
            return "Junior Software Engineer"
        elif 'principal' in title or 'staff' in title:
            return "Principal Software Engineer"
        elif 'front' in title and ('end' in title or 'developer' in title):
            return "Frontend Engineer"
        elif 'back' in title and ('end' in title or 'developer' in title):
            return "Backend Engineer"
        elif 'full' in title and ('stack' in title or 'developer' in title):
            return "Full Stack Engineer"
        else:
            return "Software Engineer"
    
    # Data Science roles
    elif re.search(r'data\s+scien|analyst', title):
        if 'senior' in title or 'sr' in title.split() or 'lead' in title:
            return "Senior Data Scientist"
        elif 'junior' in title or 'jr' in title.split():
            return "Junior Data Scientist"
        elif 'principal' in title or 'staff' in title:
            return "Principal Data Scientist"
        elif 'analyst' in title:
            return "Data Analyst"
        else:
            return "Data Scientist"
    
    # Machine Learning roles
    elif re.search(r'machine\s+learning|ml\s+engineer|ai\s+engineer|deep\s+learning', title):
        if 'senior' in title or 'sr' in title.split() or 'lead' in title:
            return "Senior ML Engineer"
        elif 'junior' in title or 'jr' in title.split():
            return "Junior ML Engineer"
        elif 'principal' in title or 'staff' in title:
            return "Principal ML Engineer"
        elif 'research' in title:
            return "ML Research Scientist"
        else:
            return "ML Engineer"
    
    # DevOps/Cloud roles
    elif re.search(r'devops|sre|site\s+reliability|infra|cloud', title):
        return "DevOps/Cloud Engineer"
    
    # Product/Project Management
    elif re.search(r'product\s+manager|project\s+manager|program\s+manager', title):
        return "Product/Project Manager"
    
    # Catch-all for other tech roles
    elif re.search(r'tech|engineer|developer|architect|administrator|it\b', title):
        return "Other Tech Role"
    
    return "Other"

def extract_experience_level(description):
    """Extract experience level from job description"""
    if not description or pd.isna(description):
        return 0
    
    description = description.lower()
    
    # Look for explicit experience requirements
    experience_patterns = [
        r'(\d+)\+?\s+years?\s+(?:of\s+)?experience',
        r'experience\s*:?\s*(\d+)\+?\s+years?',
        r'minimum\s+(?:of\s+)?(\d+)\+?\s+years?\s+(?:of\s+)?experience',
        r'at\s+least\s+(\d+)\+?\s+years?\s+(?:of\s+)?experience'
    ]
    
    for pattern in experience_patterns:
        matches = re.search(pattern, description)
        if matches:
            try:
                years = int(matches.group(1))
                return years
            except:
                pass
    
    # Look for more general indicators
    if re.search(r'\bsenior\b|\bsr\.?\b', description):
        return 5
    elif re.search(r'\bjunior\b|\bjr\.?\b|\bentry[- ]level\b', description):
        return 1
    elif re.search(r'\bintermediate\b|\bmid[- ]level\b', description):
        return 3
    elif re.search(r'\bprincipal\b|\bstaff\b|\barchitect\b', description):
        return 8
    elif re.search(r'\bdirector\b|\bvp\b|\bhead\b', description):
        return 10
    
    # Default to mid-level if no clear indicators
    return 3

def extract_top_locations(location_series):
    """Group and standardize locations"""
    # Clean and standardize locations
    cleaned_locations = location_series.fillna("Unknown").apply(lambda x: 
        re.sub(r'\s+Remote$|\(Remote\)|\bRemote\b', '', x).strip() if isinstance(x, str) else "Unknown"
    )
    
    # Count occurrences
    location_counts = Counter(cleaned_locations)
    
    # Get top locations (more than 5 occurrences)
    top_locations = {loc for loc, count in location_counts.items() if count >= 5}
    
    # Add a category for Remote
    top_locations.add("Remote")
    
    # Standardize function to map to top locations or "Other"
    def standardize_location(loc):
        if pd.isna(loc) or not isinstance(loc, str):
            return "Unknown"
            
        # Check if it's a remote position
        if 'remote' in loc.lower():
            return "Remote"
            
        # Check for common cities/areas and standardize them
        if re.search(r'new york|nyc|brooklyn|manhattan|queens|bronx', loc.lower()):
            return "New York, NY"
        elif re.search(r'san francisco|sf|bay area|oakland|berkeley', loc.lower()):
            return "San Francisco Bay Area, CA"
        elif re.search(r'los angeles|la\b|hollywood|santa monica|pasadena', loc.lower()):
            return "Los Angeles, CA"
        elif re.search(r'seattle|bellevue|redmond|tacoma', loc.lower()):
            return "Seattle, WA"
        elif re.search(r'boston|cambridge|somerville', loc.lower()):
            return "Boston, MA"
        elif re.search(r'chicago', loc.lower()):
            return "Chicago, IL"
        elif re.search(r'austin|texas', loc.lower()):
            return "Austin, TX"
        elif re.search(r'washington,?\s*d\.?c\.?|dc\b', loc.lower()):
            return "Washington, DC"
        
        # If it's in top locations, keep it
        if loc in top_locations:
            return loc
            
        # Otherwise, mark as Other
        return "Other"
    
    # Apply standardization
    return location_series.apply(standardize_location)

def preprocess_data(input_file, output_file):
    """Main preprocessing function"""
    print("Loading raw data...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} records")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['job_title', 'company', 'description'])
    print(f"After removing duplicates: {len(df)} records")
    
    # Drop records with missing essential data
    df = df.dropna(subset=['job_title', 'company'])
    print(f"After dropping records with missing titles/companies: {len(df)} records")
    
    # Standardize job titles
    print("Standardizing job titles...")
    df['standardized_title'] = df['job_title'].apply(standardize_job_title)
    
    # Clean salary data
    print("Cleaning salary data...")
    salary_data = df['salary'].apply(clean_salary)
    df['salary_low'] = [x[0] for x in salary_data]
    df['salary_avg'] = [x[1] for x in salary_data]
    
    # Normalize locations
    print("Standardizing locations...")
    df['standardized_location'] = extract_top_locations(df['location'])
    
    # Extract experience levels
    print("Extracting experience levels...")
    df['experience_years'] = df['description'].apply(extract_experience_level)
    
    # Convert skills list to individual columns
    print("Processing skills data...")
    # Ensure skills column is a list
    df['skills'] = df['skills'].apply(lambda x: 
        eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') 
        else [] if pd.isna(x) else x
    )
    
    # Get all unique skills
    all_skills = set()
    for skills_list in df['skills']:
        if isinstance(skills_list, list):
            all_skills.update(skills_list)
    
    # Add a column for each skill
    for skill in all_skills:
        df[f"skill_{skill.lower().replace(' ', '_')}"] = df['skills'].apply(
            lambda skills_list: 1 if isinstance(skills_list, list) and skill in skills_list else 0
        )
    
    # Count total skills per job
    df['skills_count'] = df['skills'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    # Create company size feature (dummy for now - in a real project we'd look up company info)
    # For now let's create a random proxy
    np.random.seed(42)  # For reproducibility
    df['company_size'] = np.random.choice(['Small', 'Medium', 'Large'], size=len(df))
    
    # Clean up and save the processed data
    print("Saving processed data...")
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")
    
    return df

# Example usage
if __name__ == "__main__":
    preprocessed_df = preprocess_data('job_data_raw.csv', 'job_data_processed.csv')
    
    # Print some stats
    print("\nData Statistics:")
    print(f"Number of unique standardized job titles: {preprocessed_df['standardized_title'].nunique()}")
    print(f"Percentage of jobs with salary info: {preprocessed_df['salary_avg'].notna().mean() * 100:.1f}%")
    print(f"Average number of skills per job: {preprocessed_df['skills_count'].mean():.1f}")
    
    # Most common job titles
    print("\nMost common job titles:")
    print(preprocessed_df['standardized_title'].value_counts().head(5))
    
    # Most common locations
    print("\nMost common locations:")
    print(preprocessed_df['standardized_location'].value_counts().head(5))