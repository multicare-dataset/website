import streamlit as st
import pandas as pd
import ast
import os
import re
from PIL import Image
from streamlit_option_menu import option_menu

# Streamlit page configuration
st.set_page_config(page_title="Multicare Dataset", page_icon=":stethoscope:", layout="wide")

# Functions to load data with caching to improve performance
@st.cache_data
def load_article_metadata(file_folder):
    """
    Load article metadata from a parquet file.
    """
    df = pd.read_parquet(os.path.join(file_folder, 'article_metadata_website_version.parquet'))
    df['year'] = df['year'].astype(int)
    return df

@st.cache_data
def load_image_metadata(file_folder):
    """
    Load image metadata from a parquet file.
    """
    df = pd.read_parquet(os.path.join(file_folder, 'image_metadata_website_version.parquet'))
    df.rename({'postprocessed_label_list': 'labels'}, axis = 1, inplace = True)
    df['labels'] = df.labels.apply(ast.literal_eval)
    return df

@st.cache_data
def load_cases(file_folder, min_year, max_year):
    """
    Load case data from multiple parquet files based on the year range.
    """
    df = pd.DataFrame()
    for file_ in ['cases_1990_2012.parquet', 'cases_2013_2017.parquet', 'cases_2018_2021.parquet', 'cases_2022_2024.parquet']:
        years = file_.split('.')[0].split('_')[1:]
        if (max_year >= int(years[0])) and (min_year <= int(years[1])):
            df = pd.concat([df, pd.read_parquet(os.path.join(file_folder, file_))], ignore_index=True)
    return df

class ClinicalCaseHub():

    def __init__(self, article_metadata_df, image_metadata_df, cases_df, image_folder='img'):
        """
        Class initialization.
        article_metadata_df (DataFrame): DataFrame containing article metadata.
        image_metadata_df (DataFrame): DataFrame containing image metadata.
        cases_df (DataFrame): DataFrame containing case data.
        image_folder (str): Folder where images are stored.
        """
        self.image_folder = image_folder

        self.full_metadata_df = article_metadata_df.copy()
        self.full_image_metadata_df = image_metadata_df.copy()
        self.full_cases_df = cases_df.copy()
        self.full_cases_df['age'] = self.full_cases_df['age'].astype(int, errors='ignore')

    def apply_filters(self, filter_dict):
        """
        Apply filters to the data based on the filter dictionary.
        filter_dict (dict): Dictionary containing filter parameters.
        """
        self.filter_dict = filter_dict

        # Filter article metadata
        self.metadata_df = self.full_metadata_df[
            (self.full_metadata_df.year >= self.filter_dict['min_year']) &
            (self.full_metadata_df.year <= self.filter_dict['max_year'])
        ].copy()
        if self.filter_dict['license'] == 'commercial':
            self.metadata_df = self.metadata_df[self.metadata_df.commercial_use_license == True]

        # Filter cases
        self.cases_df = self.full_cases_df[self.full_cases_df['article_id'].isin(self.metadata_df['article_id'])]
        if self.filter_dict['min_age'] != 0:
            self.cases_df = self.cases_df[self.cases_df.age >= self.filter_dict['min_age']]
        if self.filter_dict['max_age'] != 100:
            self.cases_df = self.cases_df[self.cases_df.age <= self.filter_dict['max_age']]
        if self.filter_dict['gender'] != 'Any':
            self.cases_df = self.cases_df[self.cases_df.gender == self.filter_dict['gender']]
        if self.filter_dict['case_search']:
            self.cases_df = self.cases_df[self.cases_df.case_text.apply(lambda x: self._text_matches_conditions(x, self.filter_dict['case_search']))]

        # Filter image metadata
        self.image_metadata_df = self.full_image_metadata_df.copy()
        self.image_metadata_df = self.image_metadata_df[self.image_metadata_df['article_id'].isin(self.metadata_df['article_id'])]
        if self.filter_dict['image_type_label']:
            self.image_metadata_df = self.image_metadata_df[self.image_metadata_df.labels.apply(lambda x: self.filter_dict['image_type_label'] in x)]
        if self.filter_dict['anatomical_region_label']:
            self.image_metadata_df = self.image_metadata_df[self.image_metadata_df.labels.apply(lambda x: self.filter_dict['anatomical_region_label'] in x)]
        if filter_dict['caption_search']:
            self.image_metadata_df = self.image_metadata_df[self.image_metadata_df.caption.apply(lambda x: self._text_matches_conditions(x, self.filter_dict['caption_search']))]

        # Harmonize data
        self.filtered_article_ids = set(self.metadata_df['article_id'].unique()) & set(self.cases_df['article_id'].unique()) & set(self.image_metadata_df['article_id'].unique())
        self.filtered_case_ids = set(self.cases_df['case_id'].unique()) & set(self.image_metadata_df['case_id'].unique())

        self.metadata_df = self.metadata_df[self.metadata_df['article_id'].isin(self.filtered_article_ids)]
        self.cases_df = self.cases_df[(self.cases_df['case_id'].isin(self.filtered_case_ids)) & (self.cases_df['article_id'].isin(self.filtered_article_ids))]
        self.image_metadata_df = self.image_metadata_df[(self.image_metadata_df['case_id'].isin(self.filtered_case_ids)) & (self.image_metadata_df['article_id'].isin(self.filtered_article_ids))]

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



# ---------- STREAMLIT CODE --------------

# Global CSS 
st.markdown(
    """
    <style>
    .stMainBlockContainer {
        padding-top: 3rem;
    }

    .stLogo {
        margin: auto;
        height: 45px;
    }

    .stFormSubmitButton, .stButton, .stDownloadButton {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    with st.sidebar:
        st.logo("multicare-logo.webp", size="large")
    
        # Define the menu options
        selected = option_menu(
            menu_title=None,
            options=["Home", "Search", "About"],
            icons=["house", "search", "info-circle"],
            menu_icon="cast",
            default_index=0,
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "nav-link-selected": {"background-color": "#12588ECC", "font-weight": 700},
            },
        )
        
        image_path = os.path.join('.', 'medical_doctor_desktop.webp')
        st.image(Image.open(image_path))

    if selected == "Home":
        st.title("Clinical Case Hub")
        st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.")


    elif selected == "Search":      

        with st.form("filter_form"):
            st.subheader("Filters")

            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Year slider: double-sided slider
                min_year, max_year = st.slider("Year", 1990, 2024, (2023, 2024))
            
                # Gender
                gender = st.selectbox("Gender", options=['Any', 'Female', 'Male'])
            
                # Image Type Label
                image_type_options = ['ct', 'mri', 'x_ray', 'ultrasound', 'angiography', 'mammography', 'echocardiogram', 'cholangiogram',
                                      'cta', 'cmr', 'mra', 'mrcp', 'spect', 'pet', 'scintigraphy', 'tractography',
                                      'skin_photograph', 'oral_photograph', 'other_medical_photograph', 'fundus_photograph', 'ophtalmic_angiography', 'oct',
                                      'pathology', 'h&e', 'immunostaining', 'immunofluorescence', 'acid_fast', 'masson_trichrome', 'giemsa', 'papanicolaou', 'gram', 'fish',
                                      'endoscopy', 'colonoscopy', 'bronchoscopy', 'ekg', 'eeg', 'chart']
                image_type_label = st.selectbox("Image Type Label", options=[''] + image_type_options)
                image_type_label = image_type_label if image_type_label != '' else None
            
            with col2:
                # Age slider: double-sided slider
                min_age, max_age = st.slider("Age", 0, 100, (15, 45))
            
                # Case Text Search
                case_search = st.text_input("Case Text Search", value='')
            
                # Anatomical Region Label
                anatomical_region_options = ['head', 'neck', 'thorax', 'abdomen', 'pelvis', 'upper_limb', 'lower_limb', 'dental_view']
                anatomical_region_label = st.selectbox("Anatomical Region Label", options=[''] + anatomical_region_options)
                anatomical_region_label = anatomical_region_label if anatomical_region_label != '' else None

                submitted = st.form_submit_button("Apply Filters")

            with col3:
                            
                # License as horizontal radio buttons
                license = st.radio("License", options=['all', 'commercial'], horizontal=True)
                
                # Caption Text Search
                caption_search = st.text_input("Caption Text Search", value='')
            
                # Resource Type, adding 'both' option
                resource = st.selectbox("Resource Type", options=['text', 'image', 'both'], index=0)

    
        
        if submitted: 
        
            # Create filter dictionary
            filter_dict = {
                'min_age': min_age,
                'max_age': max_age,
                'gender': gender,
                'case_search': case_search,
                'image_type_label': image_type_label,
                'anatomical_region_label': anatomical_region_label,
                'caption_search': caption_search,
                'min_year': min_year,
                'max_year': max_year,
                'resource': resource,
                'license': license
            }
    
            # Load data
            file_folder = '.'
            article_metadata_df = load_article_metadata(file_folder)
            image_metadata_df = load_image_metadata(file_folder)
            cases_df = load_cases(file_folder, min_year, max_year)
            
            # Instantiate the class
            cch = ClinicalCaseHub(article_metadata_df, image_metadata_df, cases_df, image_folder='img')
    
            # Apply filters
            cch.apply_filters(filter_dict)
            num_results = len(cch.cases_df)
            
            # Pagination setup
            results_per_page = 5
        
            if filter_dict['resource'] == 'text':
                num_results = len(cch.cases_df)
                st.write(f"Number of results: {num_results}")
                if num_results == 0:
                    st.write("No results found.")
                else:
                    st.write(f"Number of results: {num_results}")
                
                    # Configurar el estado de la página
                    if 'page_number' not in st.session_state:
                        st.session_state.page_number = 1
                
                    # Botones de navegación
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("Previous") and st.session_state.page_number > 1:
                            st.session_state.page_number -= 1
                    with col2:
                        st.write(f"Page {st.session_state.page_number}")
                    with col3:
                        if st.button("Next") and st.session_state.page_number * results_per_page < num_results:
                            st.session_state.page_number += 1
                
                    # Calcular índices de resultados
                    start_idx = (st.session_state.page_number - 1) * results_per_page
                    end_idx = min(start_idx + results_per_page, num_results)
                
                    for index in range(start_idx, end_idx):
                        display_case_text(cch, index)
        
            elif filter_dict['resource'] == 'image':
                num_results = len(cch.image_metadata_df)
                st.write(f"Number of results: {num_results}")
                if num_results == 0:
                    st.write("No results found.")
                else:
                    # Pagination
                    total_pages = (num_results + results_per_page - 1) // results_per_page
                    page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
                    start_idx = (page_number - 1) * results_per_page
                    end_idx = min(start_idx + results_per_page, num_results)
                    for index in range(start_idx, end_idx):
                        display_image(cch, index)
        
            elif filter_dict['resource'] == 'both':
                num_results = len(cch.cases_df)
                st.write(f"Number of results: {num_results}")
                if num_results == 0:
                    st.write("No results found.")
                else:
                    # Pagination
                    total_pages = (num_results + results_per_page - 1) // results_per_page
                    page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
                    start_idx = (page_number - 1) * results_per_page
                    end_idx = min(start_idx + results_per_page, num_results)
        
                    for index in range(start_idx, end_idx):
                        display_case_both(cch, index)

                
    elif selected == "About":
        st.title("About")
        st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.")
       
def display_case_text(cch, index):
    """
    Display text case information.
    """
    # Get data
    patient_age = cch.cases_df.age.iloc[index]
    patient_gender = cch.cases_df.gender.iloc[index]
    case_id = cch.cases_df.case_id.iloc[index]
    case_text = cch.cases_df.case_text.iloc[index]
    article_id = cch.cases_df.article_id.iloc[index]
    article_citation = cch.metadata_df[cch.metadata_df.article_id == article_id].citation.iloc[0]
    # article_link = cch.metadata_df[cch.metadata_df.article_id == article_id].link.iloc[0]

    with st.container(border=True):
        st.subheader(f"Case ID: {case_id}")
        st.write(f"Gender: {patient_gender}")
        st.write(f"Age: {patient_age}")
        
        with st.expander("Case Description"):
            st.markdown(
                f"<div style='text-align: justify; padding:2rem;'>{case_text}</div>",
                unsafe_allow_html=True
            )
            
        #  st.write(f"Article Link: [Link]({article_link})")
        st.write(f"Citation: {article_citation}")


def display_image(cch, index):
    """
    Display image and related information.
    """
    image_file = cch.image_metadata_df.file.iloc[index]
    image_path = os.path.join(cch.image_folder, image_file)
    image_caption = cch.image_metadata_df.caption.iloc[index]
    image_labels = cch.image_metadata_df.labels.iloc[index]
    case_id = cch.image_metadata_df.case_id.iloc[index]
    article_id = cch.image_metadata_df.article_id.iloc[index]

    patient_age = cch.cases_df[cch.cases_df.case_id == case_id].age.iloc[0]
    patient_gender = cch.cases_df[cch.cases_df.case_id == case_id].gender.iloc[0]

    article_citation = cch.metadata_df[cch.metadata_df.article_id == article_id].citation.iloc[0]
    article_link = cch.metadata_df[cch.metadata_df.article_id == article_id].link.iloc[0]

    with st.container(border=True):
        st.subheader(f"Case ID: {case_id}")
        st.write(f"Gender: {patient_gender}")
        st.write(f"Age: {patient_age}")

        # Display image
        st.image(Image.open(image_path), caption=image_caption)

        st.write(f"Image Labels: {', '.join(image_labels)}")
        # st.write(f"Article Link: [Link]({article_link})")
        st.write(f"Citation: {article_citation}")

def display_case_both(cch, index):
    """
    Display both text and images for a case.
    """
    # Get data
    patient_age = cch.cases_df.age.iloc[index]
    patient_gender = cch.cases_df.gender.iloc[index]
    case_id = cch.cases_df.case_id.iloc[index]
    case_text = cch.cases_df.case_text.iloc[index]
    article_id = cch.cases_df.article_id.iloc[index]
    article_citation = cch.metadata_df[cch.metadata_df.article_id == article_id].citation.iloc[0]
    article_link = cch.metadata_df[cch.metadata_df.article_id == article_id].link.iloc[0]

    with st.container(border=True):
        st.subheader(f"Case ID: {case_id}")
        st.write(f"Gender: {patient_gender}")
        st.write(f"Age: {patient_age}")
        with st.expander("Case Description"):
            st.markdown(
                f"<div style='text-align: justify; padding:2rem;'>{case_text}</div>",
                unsafe_allow_html=True
            )

        # Display images associated with this case
        images = cch.image_metadata_df[cch.image_metadata_df.case_id == case_id]
        if not images.empty:
            for idx in images.index:
                image_file = images.at[idx, 'file']
                image_path = os.path.join(cch.image_folder, image_file)
                image_caption = images.at[idx, 'caption']
                st.image(Image.open(image_path), caption=image_caption)

        # st.write(f"Article Link: [Link]({article_link})")
        st.write(f"Citation: {article_citation}")


if __name__ == '__main__':
    main()
