import pandas as pd
import ast
import os
import re

from PIL import Image
from IPython.display import display

class ClinicalCaseHub():

  def __init__(self, file_folder = '/content', image_folder = '/content/img'):

    '''Class instantiation.
    file_folder (str): folder where parquet files are stored.
    image_folder (str): folder where images are stored.
    '''

    self.file_folder = file_folder
    self.image_folder = image_folder

    # Importing article metadata and image metadata files.
    self.full_metadata_df = pd.read_parquet(os.path.join(self.file_folder, 'article_metadata_website_version.parquet'))
    self.full_metadata_df['year'] = self.full_metadata_df['year'].astype(int)
    self.full_image_metadata_df = pd.read_parquet(os.path.join(self.file_folder, 'image_metadata_website_version.parquet'))
    self.full_image_metadata_df['labels'] = self.full_image_metadata_df.labels.apply(ast.literal_eval)

  def apply_filters(self, filter_dict):

    '''Method used to apply filters on the full dataset.
    filter_dict (dict): Dictionary with the following keys:
      - 'min_age' (int): minimum age of the patient, minimum possible value is 0.
      - 'max_age' (int): maximum age of the patient, maximum possible value is 100.
      - 'gender' (str): gender of the patient, either Female, Male, or Any.
      - 'case_search' (str): search string used for case texts. For example: "(diabetes or diabetic) AND (hypertension) NOT insulin"
      - 'image_type_label' (str): label related to image type.
              - Possible labels: ['ct', 'mri', 'x_ray', 'ultrasound', 'angiography', 'mammography', 'echocardiogram', 'cholangiogram',
                                  'cta', 'cmr', 'mra', 'mrcp', 'spect', 'pet', 'scintigraphy', 'tractography',
                                  'skin_photograph', 'oral_photograph', 'other_medical_photograph', 'fundus_photograph', 'ophtalmic_angiography', 'oct',
                                  'pathology', 'h&e', 'immunostaining', 'immunofluorescence', 'acid_fast', 'masson_trichrome', 'giemsa', 'papanicolaou', 'gram', 'fish',
                                  'endoscopy', 'colonoscopy', 'bronchoscopy', 'ekg', 'eeg', 'chart']
      - 'anatomical_region_label' (str): label related to anatomical region of the image.
              - Use only if image_type_label is in ['ct', 'mri', 'x_ray', 'ultrasound', 'angiography', 'cta', 'mra', 'spect', 'pet', 'scintigraphy'].
              - Possible labels: ['head', 'neck', 'thorax', 'abdomen', 'pelvis', 'upper_limb', 'lower_limb', 'dental_view']
      - 'caption_search' (str): search string used for captions. For example: "(metastasis or metastases) AND (brain)"
      - 'min_year' (int): minimum article year, minimum possible value is 1990.
      - 'max_year' (int): maximum article year, maximum possible value is 2024.
      - 'resource' (str): type of output (either 'text' or 'image')
      - 'license' (str): article license, either 'all' or 'commercial'.


    Example of filter_dict:

    filter_dict = {
        'min_age': 18,
        'max_age': 80,
        'gender': 'Male',
        'case_search': '(diabetic or diabetes) AND (hypertension)',
        'image_type_label': 'ct',
        'anatomical_region_label': 'head',
        'caption_search': 'aneurysm',
        'min_year': 2014,
        'max_year': 2024,
        'resource': 'image',
        'license': 'all'
    }

    '''

    self.filter_dict = filter_dict

    # Importing parquet with cases. The uploaded case parquets depend on the min and max years in the filter_dict.
    self.cases_df = self._upload_case_df(self.filter_dict['min_year'], self.filter_dict['max_year'])
    self.cases_df['age'] = self.cases_df['age'].astype(int, errors = 'ignore')

    # Applying article metadata filters.
    self.metadata_df = self.full_metadata_df[self.full_metadata_df.year >= self.filter_dict['min_year']].copy() # The full_metadata_df is copied, so reupload is not necessary if filters are changed.
    self.metadata_df = self.metadata_df[self.metadata_df.year <= self.filter_dict['max_year']]
    if self.filter_dict['license'] == 'commercial':
      self.metadata_df = self.metadata_df[self.metadata_df.commercial_use_license == True]

    # Applying case filters.
    if self.filter_dict['min_age'] != 0:
      self.cases_df = self.cases_df[self.cases_df.age >= self.filter_dict['min_age']]
    if self.filter_dict['max_age'] != 100:
      self.cases_df = self.cases_df[self.cases_df.age <= self.filter_dict['max_age']]
    if self.filter_dict['gender'] != 'Any':
      self.cases_df = self.cases_df[self.cases_df.gender == self.filter_dict['gender']]
    if self.filter_dict['case_search']:
      self.cases_df = self.cases_df[self.cases_df.case_text.apply(lambda x: self._text_matches_conditions(x, self.filter_dict['case_search']))]

    # Applying image metadata filters.
    self.image_metadata_df = self.full_image_metadata_df.copy() # The full_image_metadata_df is copied, so reupload is not necessary if filters are changed.
    if self.filter_dict['image_type_label']:
      self.image_metadata_df = self.image_metadata_df[self.image_metadata_df.labels.apply(lambda x: self.filter_dict['image_type_label'] in x)]
    if self.filter_dict['anatomical_region_label']:
      self.image_metadata_df = self.image_metadata_df[self.image_metadata_df.labels.apply(lambda x: self.filter_dict['anatomical_region_label'] in x)]
    if filter_dict['caption_search']:
      self.image_metadata_df = self.image_metadata_df[self.image_metadata_df.caption.apply(lambda x: self._text_matches_conditions(x, self.filter_dict['caption_search']))]

    # Data harmonization (for instance, if an article is filtered out from one df, it should be filtered out from the rest).
    self.filtered_article_ids = set(self.metadata_df['article_id'].unique()) & set(self.cases_df['article_id'].unique()) & set(self.image_metadata_df['article_id'].unique())
    self.filtered_case_ids = set(self.cases_df['case_id'].unique()) & set(self.image_metadata_df['case_id'].unique())

    self.metadata_df = self.metadata_df[self.metadata_df['article_id'].isin(self.filtered_article_ids)]
    self.cases_df = self.cases_df[(self.cases_df['case_id'].isin(self.filtered_case_ids)) & (self.cases_df['article_id'].isin(self.filtered_article_ids))]
    self.image_metadata_df = self.image_metadata_df[(self.image_metadata_df['case_id'].isin(self.filtered_case_ids)) & (self.image_metadata_df['article_id'].isin(self.filtered_article_ids))]


  def _upload_case_df(self, min_year, max_year):

    df = pd.DataFrame()
    for file_ in ['cases_1990_2012.parquet', 'cases_2013_2017.parquet', 'cases_2018_2021.parquet', 'cases_2022_2024.parquet']:
      years = file_.split('.')[0].split('_')[1:]
      if (max_year >= int(years[0])) and (min_year <= int(years[1])):
        df = pd.concat([df, pd.read_parquet(os.path.join(self.file_folder, file_))])
    return df

  def _text_matches_conditions(self, text, query):
    """
    Checks if a given text meets all conditions defined in the parsed list.
    """

    parsed_list = self._parse_search_string(query)
    text = text.lower()  # Make text lowercase for case-insensitivity
    for condition in parsed_list:
      operator = condition['operator']
      substrings = condition['substring']
      
      if operator == "AND":
        # True if the text contains any of the substrings as full words
        if not any(self._full_word_match(text, sub) for sub in substrings):
          return False
      elif operator == "NOT":
        # False if the text contains any of the substrings as full words
        if any(self._full_word_match(text, sub) for sub in substrings):
          return False
    
    return True

  def _parse_search_string(self, query):
    """
    Parses a search string into a list of dictionaries with operators and substrings.

    '(diabetes Or diabetic) and (hypertension) NOT (stroke)' is turned into: [{'operator': 'AND', 'substring': ['diabetes', 'diabetic']}, {'operator': 'AND', 'substring': ['hypertension'], {'operator': 'NOT', 'substring': ['stroke']}}
    ]'
    """
    # Split by AND and NOT while keeping the delimiters
    tokens = re.split(r'(?<=\b)(AND|NOT)(?=\b)', query, flags=re.IGNORECASE)
    
    parsed_list = []
    current_operator = "AND"  # Default operator
    
    for token in tokens:
      token = token.strip()
      if token.upper() in ["AND", "NOT"]:
        current_operator = token.upper()
      elif token:
        # Split the substring by OR and clean it
        substrings = [sub.strip(' "').strip() for sub in token.lower().split(" or ")]
        substrings = [re.sub(r'^[^a-zA-Z0-9\s]+|[^a-zA-Z0-9\s]+$', '', substring) for substring in substrings]
        parsed_list.append({'operator': current_operator, 'substring': substrings})
    
    return parsed_list

  def _full_word_match(self, text, word):
    """
    Checks if a word matches as a full word in the text, ignoring case and boundaries.
    """
    # Use regex with word boundaries for full-word match
    return re.search(rf'\b{re.escape(word.lower())}\b', text) is not None

  def display_data(self, index = 0):
    '''
    Displays one outcome of the search, in a different format depending on the resource type (text or image).
    index (int): index of the outcome to display.
    '''

    if self.filter_dict['resource'] == 'text':
      patient_age = self.cases_df.age.iloc[index]
      patient_gender = self.cases_df.gender.iloc[index]
      case_id = self.cases_df.case_id.iloc[index]
      case_text = self.cases_df.case_text.iloc[index]
      article_id = self.cases_df.article_id.iloc[index]
      article_citation = self.metadata_df[self.metadata_df.article_id == article_id].citation.iloc[0]
      article_link = self.metadata_df[self.metadata_df.article_id == article_id].link.iloc[0]
      display_string = f"""
      Case ID: {case_id}
      Gender: {patient_gender}
      Age: {patient_age}

      {case_text}

      Link to article: {article_link}
      Citation: {article_citation}
      """
      print(display_string)
      
    elif self.filter_dict['resource'] == 'image':
      image_file = self.image_metadata_df.file.iloc[index]
      image_path = os.path.join(self.image_folder, image_file)
      image_caption = self.image_metadata_df.caption.iloc[index]
      image_labels = self.image_metadata_df.labels.iloc[index]
      case_id = self.image_metadata_df.case_id.iloc[index]
      article_id = self.image_metadata_df.article_id.iloc[index]

      patient_age = self.cases_df[self.cases_df.case_id == case_id].age.iloc[0]
      patient_gender = self.cases_df[self.cases_df.case_id == case_id].gender.iloc[0]

      article_citation = self.metadata_df[self.metadata_df.article_id == article_id].citation.iloc[0]
      article_link = self.metadata_df[self.metadata_df.article_id == article_id].link.iloc[0]
      string_start = f"""
      Case ID: {case_id}
      Gender: {patient_gender}
      Age: {patient_age}

      """
      print(string_start)

      display(Image.open(image_path))

      string_end = f"""
      Caption: {image_caption}
      Image Labels: {', '.join(image_labels)}

      Link to article: {article_link}
      Citation: {article_citation}
      """
      print(string_end)

  if __name__ == '__main__':
    cch = ClinicalCaseHub()
    filter_dict = {'min_age': 18, 'max_age': 80, 'gender': 'Male', 'case_search': '(diabetic or diabetes) AND (hypertension)', 'image_type_label': 'mri',
        'anatomical_region_label': 'head', 'caption_search': 'aneurysm', 'min_year': 2014, 'max_year': 2024, 'resource': 'image', 'license': 'all'}
    cch.apply_filters(filter_dict)
    cch.display_data(index = 0)
