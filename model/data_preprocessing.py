import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataPreprocessing:
    def __init__(self):
        self.label_encoder_primary_skill = LabelEncoder()
        self.label_encoder_city = LabelEncoder()
        print('DataPreprocessing initialized')

    def preprocess(self, df):
        '''
        Extract average salaries to two new columns salary_b2b and salary_permanent
        drop rows where both salary_b2b and salary_permanent are NaN
        drop column employment_types
        '''
        df[["salary_b2b", "salary_permanent"]] = df["employment_types"].apply(self._extract_salaries)
        df = df.dropna(subset=["salary_b2b", "salary_permanent"])
        df = df.drop(columns=['employment_types'])

        '''
        Extract skills from list of value objects in column skills
        and store them as a list of values
        '''
        df["skills"] = df["skills"].apply(self._extract_skill_names)

        '''
        Drop features that are not useful
        '''
        df = df.drop(columns=['latitude', 'longitude', 'published_at', 'id', 'company_logo_url', 'company_url', 'street', 'address_text', 'title', 'company_name', 'remote_interview', 'company_size'])

        '''
        Rename column marker_icon to primary_skill
        '''
        df = df.rename(columns={'marker_icon': 'primary_skill'})

        '''
        Apply One-Hot Encoding for categorical data
        '''
        df = pd.get_dummies(df, columns=['workplace_type', 'country_code', 'experience_level'])

        '''
        Extract values from skills column and create new numerical columns 
        for each skill value
        Drop old skills column
        Replace NaN with 0
        '''
        skills_df = df['skills'].apply(lambda x: pd.Series({skill: 1 for skill in x}))
        df = pd.concat([df, skills_df], axis=1)
        df.drop(columns=['skills'], inplace=True)
        df.fillna(0, inplace=True)

        '''
        Add Label Encoding for primary_skill and city columns
        '''
        df['primary_skill'] = self.label_encoder_primary_skill.fit_transform(df['primary_skill'])
        df['city'] = self.label_encoder_city.fit_transform(df['city'])

        return df

    def _extract_salaries(self, employment_types):
        salary_b2b = None
        salary_permanent = None
        
        if employment_types:
            for job_type in employment_types:
                if job_type and "salary" in job_type and job_type["salary"]:
                    salary_from = job_type["salary"].get("from", 0)
                    salary_to = job_type["salary"].get("to", 0)
                    avg_salary = (salary_from + salary_to) / 2
                    if job_type["type"] == "b2b":
                        salary_b2b = avg_salary
                        salary_permanent = 0
                    elif job_type["type"] == "permanent":
                        salary_permanent = avg_salary
                        salary_b2b = 0

        return pd.Series([salary_b2b, salary_permanent])
    
    def _extract_skill_names(self, skills):
        return [skill["name"] for skill in skills]
    
    def _extract_label_encoder_class(label_encoder, class_name):
        class_name = class_name.strip().lower()
        if class_name in label_encoder.classes_:
            return label_encoder.transform([class_name])[0]
        else:
            return None