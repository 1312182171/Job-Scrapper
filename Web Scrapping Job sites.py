import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from datetime import datetime
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

# Download NLTK data if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Function to clean text
def clean_text(text):
    if text:
        return ' '.join(text.split())
    return None

def setup_driver():
    """Set up and return a Selenium WebDriver instance"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    
    return webdriver.Chrome(options=chrome_options)

def get_simplyhired_url(position, location):
    """Format the SimplyHired URL"""
    position = position.replace(' ', '+')
    location = location.replace(' ', '+')
    return f"https://www.simplyhired.com/search?q={position}&l={location}"

def extract_skills(text):
    """Extract technology skills from text using NLP techniques"""
    if not text or text == "N/A":
        return []
    
    # Comprehensive list of technical skills and technologies
    tech_skills = [
        # Programming Languages
        "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "php", "swift", "kotlin", 
        "go", "rust", "scala", "perl", "r", "matlab", "dart", "groovy", "powershell", "bash",
        "shell", "c", "objective-c", "assembly", "haskell", "lua", "fortran", "cobol", "ada",
        "lisp", "prolog", "erlang", "clojure", "elixir", "f#", "vba", "delphi", "julia",
        
        # Web Development
        "html", "css", "sass", "less", "jquery", "react", "angular", "vue", "node.js", "express",
        "django", "flask", "spring", "asp.net", "laravel", "ruby on rails", "gatsby", "next.js",
        "bootstrap", "tailwind", "webpack", "babel", "redux", "graphql", "rest", "soap", "materialui",
        "wordpress", "drupal", "joomla", "magento", "shopify", "woocommerce", "ember", "svelte",
        "meteor", "backbone", "struts", "jsp", "thymeleaf", "hugo", "nuxt.js", "apollo",
        
        # Database
        "sql", "mysql", "postgresql", "mongodb", "oracle", "sqlite", "redis", "dynamodb", "cassandra",
        "mariadb", "elasticsearch", "firebase", "neo4j", "graphql", "cosmos db", "bigtable",
        "sql server", "hbase", "influxdb", "couchdb", "riak", "realm", "terrastore", "cockroachdb", 
        "sybase", "db2", "supabase", "planetscale", "memcached", "snowflake", "teradata",
        
        # Cloud & DevOps
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins", "gitlab ci", "github actions",
        "circleci", "ansible", "puppet", "chef", "prometheus", "grafana", "heroku", "serverless",
        "openstack", "vmware", "vagrant", "digitalocean", "cloudflare", "vercel", "netlify", "lambda",
        "ec2", "s3", "rds", "dynamodb", "asg", "eks", "ecs", "beanstalk", "cloudfront", "route53",
        "iam", "cloudwatch", "sqs", "sns", "azure functions", "app service", "cosmos db", "blob storage",
        
        # Data Science & ML
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "tableau", "power bi",
        "hadoop", "spark", "kafka", "airflow", "sas", "spss", "matplotlib", "seaborn", "d3.js",
        "looker", "dbt", "scipy", "statsmodels", "pyspark", "nltk", "opencv", "hugging face",
        "transformers", "mlflow", "kubeflow", "luigi", "prefect", "dataflow", "bigquery", "redshift",
        "databricks", "qlik", "dax", "powerquery", "mllib", "h2o", "weka", "rapidminer", "knime",
        
        # Mobile
        "android", "ios", "react native", "flutter", "xamarin", "ionic", "cordova", "swift", "kotlin",
        "objective-c", "swiftui", "uikit", "android studio", "xcode", "cocoapods", "gradle", 
        "firebase", "appcenter", "realm", "jetpack", "mvvm", "reactive", "rx", "compose", "unity",
        
        # Tools & Methodologies
        "git", "svn", "jira", "confluence", "trello", "agile", "scrum", "kanban", "ci/cd",
        "tdd", "rest api", "graphql", "soap", "microservices", "oauth", "jwt", "webpack",
        "bitbucket", "github", "gitlab", "azure devops", "slack", "notion", "figma", "sketch",
        "invision", "zeplin", "adobe xd", "postman", "swagger", "openapi", "sentry", "newrelic",
        
        # Other Tech
        "linux", "unix", "windows", "macos", "blockchain", "ai", "machine learning", "nlp",
        "computer vision", "iot", "ar", "vr", "embedded systems", "networking", "security", 
        "tcp/ip", "dns", "http", "ssl", "tls", "ipv4", "ipv6", "dhcp", "nginx", "apache",
        "iis", "tomcat", "websockets", "mqtt", "soa", "rest", "grpc", "protobuf", "etl",
        
        # Additional Skills
        "excel", "word", "powerpoint", "photoshop", "illustrator", "indesign", "premiere",
        "after effects", "analytics", "seo", "sem", "digital marketing", "social media",
        "content management", "project management", "pmp", "prince2", "itil", "togaf",
        
        # Soft Skills (if needed)
        "communication", "teamwork", "leadership", "problem-solving", "critical thinking",
        "time management", "adaptability", "creativity", "emotional intelligence", "negotiation"
    ]
    
    # Make text lowercase for comparison
    text_lower = text.lower()
    
    # Find all occurrences of skills in the text
    found_skills = []
    
    for skill in tech_skills:
        # Use word boundary to match whole words
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    
    # Look for additional potential skills (capitalize technical terms)
    words = word_tokenize(text_lower)
    potential_skills = []
    
    # Look for capitalized words, acronyms, or words connected to technical terms
    tech_indicators = ["software", "framework", "library", "platform", "tools", "system", 
                       "programming", "language", "develop", "code", "tech", "technology"]
    
    for i, word in enumerate(words):
        # Check for acronyms (all caps)
        if len(word) >= 2 and word.isupper() and word.isalpha():
            potential_skills.append(word.lower())
        
        # Check for CamelCase or capitalized words that might be technologies
        elif len(word) >= 2 and any(c.isupper() for c in word[1:]) and word not in found_skills:
            potential_skills.append(word.lower())
        
        # Check for words that follow technical indicators
        elif i > 0 and words[i-1].lower() in tech_indicators and word not in found_skills:
            potential_skills.append(word.lower())
    
    # Combine and remove duplicates
    all_skills = list(set(found_skills + potential_skills))
    
    return all_skills

class JobScraper:
    def __init__(self):
        self.results = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def scrape_simplyhired_selenium(self, query="software+engineer", location="New+York%2C+NY", pages=3):
        """Scrape SimplyHired using Selenium for better handling of dynamic content"""
        print("Scraping SimplyHired with Selenium...")
        
        for page in range(1, pages+1):
            url = f"https://www.simplyhired.com/search?q={query}&l={location}&pn={page}"
            print(f"Scraping page {page}: {url}")
            
            driver = setup_driver()
            
            try:
                driver.get(url)
                
                # Wait for job cards to load
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='searchSerpJob']"))
                    )
                except TimeoutException:
                    print("Timed out waiting for SimplyHired job cards to load")
                    continue
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                
                # Based on the HTML structure, the job cards are in li.css-0 elements
                cards = soup.select('li.css-0')
                
                if not cards:
                    print("No SimplyHired job cards found with known selectors. Trying backup method...")
                    # Backup: look for any job listings
                    cards = soup.select('div[data-testid="searchSerpJob"]')
                
                print(f"Found {len(cards)} jobs on page {page}")
                
                for card in cards:
                    try:
                        # Extract job details based on the provided HTML structure
                        job_title_elem = card.select_one('h2[data-testid="searchSerpJobTitle"] a')
                        job_title = job_title_elem.text.strip() if job_title_elem else "N/A"
                        
                        # Get job URL
                        if job_title_elem and job_title_elem.get('href'):
                            href = job_title_elem.get('href')
                            job_url = f"https://www.simplyhired.com{href}" if href.startswith('/') else href
                        else:
                            job_url = "N/A"
                        
                        # Get company name
                        company_elem = card.select_one('span[data-testid="companyName"]')
                        company = company_elem.text.strip() if company_elem else "N/A"
                        
                        # Get location
                        location_elem = card.select_one('span[data-testid="searchSerpJobLocation"]')
                        location = location_elem.text.strip() if location_elem else "N/A"
                        
                        # Get job summary
                        summary_elem = card.select_one('p[data-testid="searchSerpJobSnippet"]')
                        summary = summary_elem.text.strip() if summary_elem else "N/A"
                        
                        # Extract all possible skills from job title and summary
                        all_text = f"{job_title} {summary}"
                        skills = extract_skills(all_text)
                        
                        # Get posted date
                        date_elem = card.select_one('p[data-testid="searchSerpJobDateStamp"]')
                        posted_date = date_elem.text.strip() if date_elem else "N/A"
                        
                        # Get salary - uses data-testid="searchSerpJobSalaryConfirmed"
                        salary_elem = card.select_one('p[data-testid="searchSerpJobSalaryConfirmed"]')
                        salary = salary_elem.text.strip() if salary_elem else 'N/A'
                        
                        # Get job ID from data-jobkey attribute
                        job_div = card.select_one('div[data-jobkey]')
                        job_id = job_div.get('data-jobkey') if job_div else 'N/A'
                        
                        self.results.append({
                            "source": "SimplyHired",
                            "title": job_title,
                            "company": company,
                            "location": location,
                            "skills": skills,
                            "salary": salary,
                            "url": job_url,
                            "posted_date": posted_date,
                            "job_id": job_id,
                            "summary": summary
                        })
                        
                    except Exception as e:
                        print(f"Error extracting SimplyHired job card: {e}")
                
                # Add delay to avoid overloading the server
                time.sleep(2)
                
            except Exception as e:
                print(f"Error scraping SimplyHired page {page}: {e}")
            finally:
                driver.quit()
        
        print(f"Total jobs fetched from SimplyHired: {len([job for job in self.results if job['source'] == 'SimplyHired'])}")
    
    def fetch_usajobs(self, keyword="software engineer", location="New York", results_per_page=25, page_count=3):
        """Fetch jobs from USAJobs API"""
        print("Fetching from USAJobs API...")
        
        # REPLACE THESE WITH YOUR ACTUAL API CREDENTIALS
        api_key = "tqqZLDx28XNXIAVYplr2hlR5X5RoJ/s01KF9RSCUNTk="  # Replace with your actual API key from USAJobs
        email = "legendarymartin12138@gmail.com"  # Replace with your email used to register
            
        headers = {
            "Host": "data.usajobs.gov",
            "User-Agent": email,
            "Authorization-Key": api_key
        }
        
        base_url = "https://data.usajobs.gov/api/search"
        
        total_jobs = 0
        
        for page in range(1, page_count+1):
            print(f"Fetching page {page} from USAJobs API")
            
            params = {
                "Keyword": keyword,
                "LocationName": location,
                "ResultsPerPage": str(results_per_page),
                "Page": str(page)}
            
            try:
                response = requests.get(base_url, headers=headers, params=params)
                
                # Check if the request was successful
                if response.status_code != 200:
                    print(f"Error: USAJobs API returned status code {response.status_code}")
                    print(f"Response: {response.text}")
                    continue
                
                data = response.json()
                
                # Check if we got valid data
                if "SearchResult" not in data:
                    print(f"Error: Unexpected API response format: {data}")
                    continue
                
                search_result = data["SearchResult"]
                jobs = search_result.get("SearchResultItems", [])
                
                print(f"Found {len(jobs)} jobs on page {page}")
                total_jobs += len(jobs)
                
                for job in jobs:
                    try:
                        position = job.get("MatchedObjectDescriptor", {})
                        
                        title = position.get("PositionTitle")
                        org_name = position.get("OrganizationName")
                        company = f"{org_name} (U.S. Federal Government)"
                        
                        # Get location
                        position_locations = position.get("PositionLocation", [])
                        location = None
                        if position_locations:
                            location_info = position_locations[0]
                            city = location_info.get("LocationCity", "")
                            state = location_info.get("LocationState", "")
                            location = f"{city}, {state}" if city and state else None
                        
                        # Get salary
                        remuneration = position.get("PositionRemuneration", [{}])
                        salary = None
                        if remuneration:
                            salary_min = remuneration[0].get("MinimumRange")
                            salary_max = remuneration[0].get("MaximumRange")
                            salary_rate = remuneration[0].get("RateIntervalCode")
                            
                            if salary_min and salary_max:
                                salary = f"${salary_min} - ${salary_max} {salary_rate}"
                        
                        # Get job description and qualification summary
                        job_summary = position.get("QualificationSummary", "")
                        job_duties = position.get("UserArea", {}).get("Details", {}).get("JobSummary", "")
                        combined_text = f"{title} {job_summary} {job_duties}"
                        
                        # Extract all possible skills from text using enhanced extraction
                        skills = extract_skills(combined_text)
                        
                        # Get job URL
                        apply_uri = position.get("ApplyURI", [])
                        url = apply_uri[0] if apply_uri else None
                        
                        self.results.append({
                            "source": "USAJobs",
                            "title": title,
                            "company": company,
                            "location": location,
                            "skills": skills,
                            "salary": salary,
                            "url": url,
                            "posted_date": position.get("PublicationStartDate", "N/A"),
                            "job_id": position.get("PositionID", "N/A"),
                            "summary": job_summary[:200] + "..." if job_summary else "N/A"
                        })
                        
                    except Exception as e:
                        print(f"Error processing USAJobs listing: {e}")
                
                # Add delay between API calls
                time.sleep(2)
                
            except Exception as e:
                print(f"Error fetching USAJobs page {page}: {e}")
        
        print(f"Total jobs fetched from USAJobs: {len([job for job in self.results if job['source'] == 'USAJobs'])}")

    def save_to_csv(self, filename="job_results.csv"):
        """Save results to CSV file"""
        if not self.results:
            print("No results to save")
            return
        
        df = pd.DataFrame(self.results)
        
        # Convert skills list to comma-separated string
        df['skills'] = df['skills'].apply(lambda x: ', '.join(x) if x else '')
        
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    
    def save_to_json(self, filename="job_results.json"):
        """Save results to JSON file"""
        if not self.results:
            print("No results to save")
            return
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"Results saved to {filename}")
    
    def analyze_skills(self):
        """Analyze the most common skills in the scraped jobs"""
        if not self.results:
            print("No results to analyze")
            return {}
        
        # Extract all skills from all jobs
        all_skills = []
        for job in self.results:
            all_skills.extend(job['skills'])
        
        # Count occurrences of each skill
        skill_counter = Counter(all_skills)
        
        # Get top skills
        top_skills = skill_counter.most_common(20)
        
        print("Top skills in job listings:")
        for skill, count in top_skills:
            print(f"{skill}: {count}")
        
        return skill_counter
    
    def analyze_by_source(self):
        """Analyze jobs by source"""
        if not self.results:
            print("No results to analyze")
            return
        
        sources = {}
        for job in self.results:
            source = job['source']
            if source not in sources:
                sources[source] = 0
            sources[source] += 1
        
        print("Jobs by source:")
        for source, count in sources.items():
            print(f"{source}: {count}")

    def analyze_salary_ranges(self):
        """Analyze salary ranges in the scraped jobs"""
        if not self.results:
            print("No results to analyze")
            return
        
        # Count jobs with and without salary info
        jobs_with_salary = [job for job in self.results if job.get('salary') and job.get('salary') != 'N/A']
        
        print(f"Jobs with salary information: {len(jobs_with_salary)} out of {len(self.results)} ({len(jobs_with_salary)/len(self.results)*100:.1f}%)")
        
        # Basic salary analysis - just count ranges for now
        # A more sophisticated analysis would parse the salary strings and compute statistics
        if jobs_with_salary:
            print("Sample of salaries found:")
            for i, job in enumerate(jobs_with_salary[:10]):  # Show only first 10
                print(f"  {job['title']} at {job['company']}: {job['salary']}")

def main():
    """Main function to run the scraper"""
    scraper = JobScraper()
    
    # Get user input
    job_title = input("Enter job title (e.g., 'software engineer'): ").strip()
    location = input("Enter location (e.g., 'New York, NY'): ").strip()
    
    # Format query for URLs
    job_title_query = job_title.replace(' ', '+')
    location_query = location.replace(' ', '+')
    
    # Run scrapers
    scraper.scrape_simplyhired_selenium(query=job_title_query, location=location_query, pages=3)
    scraper.fetch_usajobs(keyword=job_title, location=location, page_count=3)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"job_results_{timestamp}.csv"
    json_filename = f"job_results_{timestamp}.json"
    
    scraper.save_to_csv(csv_filename)
    scraper.save_to_json(json_filename)
    
    # Analyze results
    scraper.analyze_skills()
    scraper.analyze_by_source()
    scraper.analyze_salary_ranges()
    
    print(f"Found {len(scraper.results)} jobs in total")

if __name__ == "__main__":
    main()