import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import joblib
import os

class DataPreprocessing:
    def __init__(self):
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.label_encoder_primary_skill = LabelEncoder()
        self.label_encoder_city = LabelEncoder()
        self.programming_languages = {
            "Python",
            "PyTorch",
            "Tensorflow",
            "Pandas",
            "Numpy",
            "Java",
            "Spring",
            "Spring Boot",
            "JavaScript",
            "C",
            "C++",
            "C#",
            ".NET",
            "Ruby",
            "Go",
            "PHP",
            "Swift",
            "Kotlin",
            "TypeScript",
            "Angular",
            "React",
            "Vue",
            "Rust",
            "SQL",
            "R",
            "Scala",
            "Objective-C",
            "Perl",
            "Shell",
            "MATLAB",
            "Dart",
            "VHDL",
            "Lua",
            "Haskell",
            "Elixir",
            "Clojure"
        }
        print('DataPreprocessing initialized')

    def preprocess(self, df):
        '''
        Drop features that are not useful
        '''
        df = df.drop(
            columns=['open_to_hire_ukrainians', 'remote', 'country_code', 'latitude', 'longitude', 'published_at', 'id', 'company_logo_url', 'company_url', 'street', 'address_text', 'title', 'company_name', 'remote_interview', 'company_size'],
            errors='ignore')

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
        Extract values from skills column and create new numerical columns 
        for each skill value
        Drop old skills column
        Replace NaN with 0
        '''
        skills_df = df['skills'].apply(lambda x: pd.Series({skill: 1 for skill in x if skill in self.programming_languages}))
        
        # Add missing skill columns for all programming languages
        for skill in self.programming_languages:
            if skill not in skills_df.columns:
                skills_df[skill] = 0  # Add the missing skill column with all 0 values

        df.drop(columns=['skills'], inplace=True)
        df = pd.concat([df, skills_df], axis=1)

        '''
        Rename column marker_icon to primary_skill
        '''
        df = df.rename(columns={'marker_icon': 'primary_skill'})

        '''
        Apply One-Hot Encoding for categorical data
        '''
        df = pd.get_dummies(df, columns=['workplace_type', 'experience_level'])

        '''
        Add Label Encoding for primary_skill and city columns
        '''
        df['primary_skill'] = self.label_encoder_primary_skill.fit_transform(df['primary_skill'])
        df['city'] = self.label_encoder_city.fit_transform(df['city'])

        required_columns = [
            'city', 'primary_skill', 'salary_b2b', 'salary_permanent',
            'Java', 'TypeScript', 'Angular', 'Ruby', 'Spring Boot', 'Python',
            'Kotlin', 'JavaScript', 'Spring', 'C#', 'PHP', 'Swift', 'SQL', 'React',
            'Objective-C', 'Rust', 'C++', 'C', 'Scala', 'Elixir', 'Go', 'PyTorch',
            'R', 'Haskell', 'Vue', 'Perl', 'Dart', 'Pandas', 'Shell', 'Numpy',
            'VHDL', 'Clojure', 'MATLAB', 'Lua', '.NET', 'Tensorflow',
            'workplace_type_office', 'workplace_type_partly_remote',
            'workplace_type_remote', 'experience_level_junior', 'experience_level_mid', 'experience_level_senior'
        ]
        
        df = df[required_columns]
        
        # Add missing columns with default value 0
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0

        df.fillna(0, inplace=True)
        print(df.columns)
        return df

    def _extract_salaries(self, employment_types):
        salary_b2b = None
        salary_permanent = None

        if employment_types:
            b2b_found = False
            permanent_found = False
            
            for job_type in employment_types:
                if job_type and "salary" in job_type and job_type["salary"]:
                    salary_from = job_type["salary"].get("from", 0)
                    salary_to = job_type["salary"].get("to", 0)
                    avg_salary = (salary_from + salary_to) / 2

                    if job_type["type"] == "b2b":
                        salary_b2b = avg_salary
                        salary_permanent = 0
                        b2b_found = True

                    elif job_type["type"] == "permanent":
                        salary_permanent = avg_salary
                        salary_b2b = 0
                        permanent_found = True

            if b2b_found and not permanent_found:
                salary_permanent = 0.8 * salary_b2b

            if permanent_found and not b2b_found:
                salary_b2b = 1.2 * salary_permanent

        return pd.Series([salary_b2b, salary_permanent])
    
    def _extract_skill_names(self, skills):
        '''
        Extract skill names, but filter out only relevant programming languages
        '''
        return [skill["name"] for skill in skills if skill["name"] in self.programming_languages]
    
    
    def _extract_label_encoder_class(label_encoder, class_name):
        class_name = class_name.strip().lower()
        if class_name in label_encoder.classes_:
            return label_encoder.transform([class_name])[0]
        else:
            return None
        
    def scale_data(self, X_train, X_test, y_train, y_test, model_name):
        '''
        Scale feature values
        '''
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)

        '''
        Scale target variables' values
        '''
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)

        self._save_scalers(model_name)
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

    def _save_scalers(self, model_name):
        model_dir = os.path.join('saved_models', model_name)
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(self.scaler_X, os.path.join(model_dir, 'scaler_X.pkl'))
        joblib.dump(self.scaler_y, os.path.join(model_dir, 'scaler_y.pkl'))
        joblib.dump(self.label_encoder_primary_skill, os.path.join(model_dir, 'label_encoder_primary_skill.pkl'))
        joblib.dump(self.label_encoder_city, os.path.join(model_dir, 'label_encoder_city.pkl'))