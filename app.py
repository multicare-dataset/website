import streamlit as st
import pandas as pd
import ast
import os
import re
from streamlit_option_menu import option_menu
import psutil

# Streamlit page configuration
st.set_page_config(page_title="Clinical Case Hub", page_icon=":stethoscope:", layout="wide")

label_dict = {
    'ct': 'CT scan',
     'mri': 'MRI',
     'x_ray': 'X-ray',
     'ultrasound': 'Ultrasound',
     'angiography': 'Angiography',
     'mammography': 'Mammography',
     'echocardiogram': 'Echocardiogram',
     'cholangiogram': 'Cholangiogram',
     'nuclear_medicine': 'Nuclear Medicine',
     'skin_photograph': 'Skin Photograph',
     'oral_photograph': 'Oral Photograph',
     'fundus_photograph': 'Fundus Photograph',
     'ophthalmic_angiography': 'Ophthalmic Angiography',
     'oct': 'Optical Coherence Tomography',
     'pathology': 'Pathology',
     'h&e': 'H&E',
     'immunostaining': 'Immunostaining',
     'immunofluorescence': 'Immunofluorescence',
     'fish': 'Fluorescence In Situ Hybridization',
     'endoscopy': 'Endoscopy',
     'ekg': 'EKG'
}

@st.cache_data
def load_article_metadata(file_folder):
    df = pd.read_parquet(os.path.join(file_folder, 'article_metadata_website_version.parquet'))
    df['year'] = df['year'].astype(int)
    return df

@st.cache_data
def load_image_metadata(file_folder):
    df = pd.read_parquet(os.path.join(file_folder, 'image_metadata_website_version.parquet'))
    df.rename({'postprocessed_label_list': 'labels'}, axis = 1, inplace = True)
    df['labels'] = df.labels.apply(ast.literal_eval)
    return df

@st.cache_data
def load_cases(file_folder, min_year, max_year):
    df = pd.DataFrame()
    for file_ in ['cases_1990_2012.parquet', 'cases_2013_2017.parquet', 'cases_2018_2021.parquet', 'cases_2022_2024.parquet']:
        years = file_.split('.')[0].split('_')[1:]
        if (max_year >= int(years[0])) and (min_year <= int(years[1])):
            df = pd.concat([df, pd.read_parquet(os.path.join(file_folder, file_))], ignore_index=True)
    return df

class ClinicalCaseHub():
    def __init__(self, article_metadata_df, image_metadata_df, cases_df, image_folder='img'):
        self.image_folder = image_folder
        self.full_metadata_df = article_metadata_df.copy()
        self.full_image_metadata_df = image_metadata_df.copy()
        self.full_cases_df = cases_df.copy()
        self.full_cases_df['age'] = self.full_cases_df['age'].astype(int, errors='ignore')

    def apply_filters(self, filter_dict):
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
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    
    .centered-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
    }

    div[role="radiogroup"][aria-label="License"] {
        margin-bottom: 0.9rem;
    }

    div[data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        color: rgb(49, 51, 63);
        text-align: justify; 
        padding:3rem;

    .stForm, .stVerticalBlockBorderWrapper {
        padding: 3rem;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)




def main():
    with st.sidebar:
        st.logo("multicare-logo.webp", size="large")
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

        st.image('medical_doctor_desktop.webp')
        st.header("Resource Usage")
        st.write(f"Memory Usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
        st.write(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")


    if selected == "Home":
        st.title("The Clinical Case Hub")
        st.write(
            """
            Welcome to The Clinical Case Hub, a platform designed to empower healthcare professionals and medical 
            students with real-world clinical cases. Our mission is to provide you with a diverse collection of 
            cases and images sourced from PubMed Central case reports, enabling you to enhance your diagnostic, 
            clinical decision-making, and critical thinking skills.
            """
        )
        st.markdown("##")
        start_button = st.button("Start your search  →")
        if start_button:
            selected == "Search"
            

    elif selected == "Search":
        st.title("The Clinical Case Hub")
        st.write(
            """
            Refine your search with filters to find the clinical cases that align with your research focus or 
            learning goals. You can filter by different criteria, such as age, gender or content of the clinical 
            case text. Select a resource type based on your interests—text, image, or both.
            """
        )
        st.markdown("##")
        with st.form("filter_form"):
            st.subheader("Filters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_year, max_year = st.slider("Year", 1990, 2024, (2014, 2024))
                resource = st.selectbox("Resource Type", options=['text', 'image', 'both'], index=0)
                image_type_label = st.selectbox("Image Type Label", options=[None] + list(label_dict.values()))
                # if image_type_label:
                #     image_type_label = [key for key, value in label_dict.items() if value == image_type_label][0]
                # else:
                #     image_type_label = None
                if image_type_label is not None:
                    image_type_label = next((key for key, value in label_dict.items() if value == image_type_label), None)
            
            with col2:
                min_age, max_age = st.slider("Age", 0, 100, (18, 65))
                gender = st.selectbox("Gender", options=['Any', 'Female', 'Male'], index=0)
                caption_search = st.text_input(
                    "Caption Text Search", value='', 
                    help="Search operators: 'OR', 'AND', 'NOT'. Groups of terms that refer to the same concept should be concatenated using 'OR'. Use 'AND' to include a new term or group of terms, and use 'NOT' to exclude them. For example: '(CT OR tomography) AND (chest OR thorax) NOT abdomen' will return chest CT scans with no mentions of the word 'abdomen'."
                )

            with col3:
                license = st.radio("License", options=['all', 'commercial'], horizontal=True, index=0)
                case_search = st.text_input(
                    "Case Text Search", 
                    value='', 
                    help="Search operators: 'OR', 'AND', 'NOT'. Groups of terms that refer to the same concept should be concatenated using 'OR'. Use 'AND' to include a new term or group of terms, and use 'NOT' to exclude them. For example: '(CT OR tomography) AND (chest OR thorax) NOT abdomen' will return chest CT scans with no mentions of the word 'abdomen'."
                )
                anatomical_region_label = st.selectbox(
                    "Anatomical Region Label", 
                    options=[None] + ['head', 'neck', 'thorax', 'abdomen', 'pelvis', 'upper_limb', 'lower_limb'],
                    help="This filter can only be combined with specific image types: 'CT scan,' 'MRI,' 'X-ray,' 'Ultrasound,' 'Angiography,' and 'Nuclear Medicine'."
                )
                
            submitted = st.form_submit_button("Search")

        if submitted: 
            st.subheader("Seach Results")
            # Pagination setup
            filter_dict = {
                'min_age': min_age, 'max_age': max_age, 'gender': gender, 'case_search': case_search,
                'image_type_label': image_type_label, 'anatomical_region_label': anatomical_region_label,
                'caption_search': caption_search, 'min_year': min_year, 'max_year': max_year,
                'resource': resource, 'license': license
            }

            # Load data
            article_metadata_df = load_article_metadata('.')
            image_metadata_df = load_image_metadata('.')
            cases_df = load_cases('.', min_year, max_year)

            # Process data
            cch = ClinicalCaseHub(article_metadata_df, image_metadata_df, cases_df)
            cch.apply_filters(filter_dict)

            st.session_state.filter_dict = filter_dict
            st.session_state.cch = cch
            st.session_state.num_results = len(cch.cases_df)

        if "cch" in st.session_state:
            cch = st.session_state.cch
            num_results = st.session_state.num_results
            results_per_page = 10
            total_pages = (num_results + results_per_page - 1) // results_per_page

            if "page_number" not in st.session_state:
                st.session_state.page_number = 1
            page_number = st.session_state.page_number
            start_idx = (page_number - 1) * results_per_page
            end_idx = min(start_idx + results_per_page, num_results)



            if st.session_state.filter_dict['resource'] == 'text':
                for index in range(start_idx, end_idx):
                    display_case_text(cch, index)
            elif st.session_state.filter_dict['resource'] == 'image':
                for index in range(start_idx, end_idx):
                    display_image(cch, index)
            else:
                for index in range(start_idx, end_idx):
                    display_case_both(cch, index)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⏮  Previous") and st.session_state.page_number > 1:
                    st.session_state.page_number -= 1
            with col2:
                    st.write(f"Displaying page {page_number} of {total_pages}")
            with col3:
                if st.button("Next  ⏭") and st.session_state.page_number < total_pages:
                    st.session_state.page_number += 1

    
    elif selected == "About":
        st.title("About the MultiCaRe Dataset")
        st.write(
            """
            MultiCaRe is a dataset containing clinical cases, labeled images, and captions, 
            all extracted from open-access case reports in PubMed Central. It is derived from over 85,000 
            case report articles, involving more than 320,000 authors and 110,000 patients. The dataset 
            is designed for healthcare professionals, medical students, and data scientists.
            """
        )
        st.subheader("Useful Links")
        st.write(
            """
            - GitHub Repository: (...) [link]
            - Zenodo Data Repository: (...) [link]
            - Image Classification Model: [https://huggingface.co/mauro-nievoff/MultiCaReClassifier]
            - Taxonomy Documentation: (...) [link]
            """
        )
        st.subheader("Our Team")
        st.write(
            """
            - Mauro Nievas Offidani, MD, MSc (https://www.linkedin.com/in/mauronievasoffidani/): Data Curation
            - María Carolina González Galtier, MD, MA (https://www.linkedin.com/in/carogaltier/): Web Development
            - Miguel Massiris (...): Web Development
            - Facundo Roffet (...): ML Model Development
            - Claudio Delrieux, PhD (...): Project Direction
            """
        )

def display_case_text(cch, index):
    patient_age = int(cch.cases_df.age.iloc[index])
    patient_gender = cch.cases_df.gender.iloc[index]
    case_id = cch.cases_df.case_id.iloc[index]
    case_text = cch.cases_df.case_text.iloc[index]
    article_id = cch.cases_df.article_id.iloc[index]
    article_citation = cch.metadata_df[cch.metadata_df.article_id == article_id].citation.iloc[0]
    article_title = cch.metadata_df[cch.metadata_df.article_id == article_id].title.iloc[0]

    with st.container(border=True):
        st.subheader("Title Here")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.write(f"Case ID: **{case_id}**")
        with col2:  
            st.write(f"Gender: **{patient_gender}**")
        with col3:  
            st.write(f"Age: **{patient_age}**")
        st.write(f"**Citation**: *{article_citation}*")
        
        max_characters = 250
        case_text_aux = case_text[:max_characters]  
        match = re.search(r'\.', case_text[max_characters:])
        if match:
            case_text_aux += case_text[max_characters:max_characters + match.start() + 1]

        rest_text = case_text[len(case_text_aux):]

        st.subheader("Case Description")
        with st.expander(f"{case_text_aux}"):
            st.markdown(
                f"<div style='text-align: justify; padding:3rem;'>{rest_text}</div>",
                unsafe_allow_html=True
            )



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

    with st.container():
        st.subheader(f"Case ID: {case_id}")
        st.write(f"Gender: {patient_gender}")
        st.write(f"Age: {patient_age}")
    
        # Center and display the image with adjusted size
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/jpeg;base64,{convert_image_to_base64(image_path)}" alt="{image_caption}" style="width: 35%; border-radius: 8px;">
                <p><em>{image_caption}</em></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.write(f"Image Labels: {', '.join(image_labels)}")
        # st.write(f"Article Link: [Link]({article_link})")
        st.write(f"Citation: {article_citation}")

def convert_image_to_base64(image_path):
    """
    Convert an image file to a base64 encoded string for HTML rendering.
    """
    from base64 import b64encode
    with open(image_path, "rb") as img_file:
        return b64encode(img_file.read()).decode("utf-8")

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
    image_labels = cch.image_metadata_df.labels.iloc[index]
    article_citation = cch.metadata_df[cch.metadata_df.article_id == article_id].citation.iloc[0]
    article_link = cch.metadata_df[cch.metadata_df.article_id == article_id].link.iloc[0]

    with st.container(border=True):
        st.subheader(f"Case ID: {case_id}")
        st.write(f"Gender: {patient_gender}")
        st.write(f"Age: {patient_age}")
        # Limitar la cantidad de caracteres iniciales
        max_characters = 350
        case_text_aux = case_text[:max_characters]
        
        # Buscar el primer punto (.) después de los caracteres iniciales
        match = re.search(r'\.', case_text[max_characters:])
        if match:
            # Extender el texto hasta el primer punto encontrado
            case_text_aux += case_text[max_characters:max_characters + match.start() + 1]
        
        with st.expander("Case Description"):
            st.markdown(
                f"<div style='text-align: justify; padding:2rem;'>{case_text_aux}</div>",
                unsafe_allow_html=True
            )

        # Display images associated with this case
        images = cch.image_metadata_df[cch.image_metadata_df.case_id == case_id]
        if not images.empty:
            for idx in images.index:
                image_file = images.at[idx, 'file']
                image_path = os.path.join(cch.image_folder, image_file)
                image_caption = images.at[idx, 'caption']
                #st.image(Image.open(image_path), caption=image_caption)
                # Center and display the image with adjusted size
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <img src="data:image/jpeg;base64,{convert_image_to_base64(image_path)}" alt="{image_caption}" style="width: 35%; border-radius: 8px;">
                        <p><em>{image_caption}</em></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
        st.write(f"Image Labels: {', '.join(image_labels)}")
        # st.write(f"Article Link: [Link]({article_link})")
        st.write(f"Citation: {article_citation}")

if __name__ == '__main__':
    main()




