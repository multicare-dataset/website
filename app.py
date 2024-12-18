import streamlit as st
import pandas as pd
import os
import re
from streamlit_option_menu import option_menu
import psutil

# Streamlit page configuration
st.set_page_config(page_title="Clinical Case Hub", page_icon=":stethoscope:", layout="wide")

# Global CSS 
st.markdown(
    """
    <style>

    div.stExpander + div.stElementContainer .stMarkdown {
        padding: 1rem;
    }
        
    h3 {
        padding: 1rem;
    }

    .stForm {
        padding: 1rem;
        margin-top: 1rem;
    }

    .stVerticalBlock {
        gap: 1rem;
    }
    
    p {
        text-align: justify; 
    }
    
    .stMainBlockContainer {
        padding-top: 3rem;
    }

    .stLogo {
        margin: 1rem auto;
        height: 34px;
    }

    .stFormSubmitButton, .stButton, .stDownloadButton {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    .stFormSubmitButton button, .stButton button, .stDownloadButton button {
        padding: 0.5rem 1rem;
        background: rgba(18, 88, 142, 0.8);
        color: #fff;
    }

    .stFormSubmitButton button:hover, .stFormSubmitButton button:active, 
    .stFormSubmitButton button:focus, .stButton button:hover, 
    .stButton button:active, .stButton button:focus,
    .stDownloadButton button:hover, .stDownloadButton button:active,
    .stDownloadButton button:focus {
        padding: 0.5rem 1rem;
        background: rgba(18, 88, 142, 0.6);
        color: #fff;
    }
    
    .centered-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
    }

    .stElementContainer:has(img) {
        display: flex;
        justify-content: center;
    }
    
    div[role="radiogroup"][aria-label="License"] {
        margin-bottom: 0.9rem;
    }

    .stExpander{
        margin: 0 1rem;
    }

    .stExpander details {
        padding-bottom: 2rem;
    }

    .stExpander details summary {
        padding-bottom: 0rem;
    }

    .stExpander details summary span {
        padding-right: 2rem;
        padding-left: 3rem;
        padding-top: 2rem;
    }

    div[data-testid="stExpanderDetails"] {
        padding-right: 3rem;
        padding-left: 3rem;
        padding-top: 1rem;
    }

    details summary span [data-testid="stMarkdownContainer"] {
        display: flex;
        flex-direction: column;
    }
    
    details summary span [data-testid="stMarkdownContainer"] p:first-of-type {
        grid-column: span 3; 
        font-size: 20px; 
        padding-bottom: 1rem; 
        text-align: justify; 
    }
    
    details summary span [data-testid="stMarkdownContainer"] p:nth-of-type(2) {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem; 
        font-size: 16px;
        text-align: left;
    }

    details summary span [data-testid="stMarkdownContainer"] p:nth-of-type(2) em:nth-child(1),
    details summary span [data-testid="stMarkdownContainer"] p:nth-of-type(2) strong:nth-child(1) {
        grid-column: 1; 
    }
    
    details summary span [data-testid="stMarkdownContainer"] p:nth-of-type(2) em:nth-child(2),
    details summary span [data-testid="stMarkdownContainer"] p:nth-of-type(2) strong:nth-child(2) {
        grid-column: 2; 
    }
    
    details summary span [data-testid="stMarkdownContainer"] p:nth-of-type(2) em:nth-child(3),
    details summary span [data-testid="stMarkdownContainer"] p:nth-of-type(2) strong:nth-child(3) {
        grid-column: 3;
    }

    details summary span [data-testid="stMarkdownContainer"] p:nth-of-type(2) strong,
    details summary span [data-testid="stMarkdownContainer"] p:nth-of-type(2) em {
        display: inline;
    }

    .stMainBlockContainer  div[data-testid="stVerticalBlockBorderWrapper"] .st-emotion-cache-1wmy9hl .e1f1d6gn1 {
        margin: 1rem;
    }

    stForm .stMainBlockContainer  div[data-testid="stVerticalBlockBorderWrapper"] .st-emotion-cache-1wmy9hl .e1f1d6gn1 {
        margin: 0rem;
    }

    .stExpander .st-emotion-cache-1wmy9hl .e1f1d6gn1 {
        margin: 0rem;
    }

    .stMainBlockContainer div[data-testid="stVerticalBlockBorderWrapper"] .st-emotion-cache-1wmy9hl .e1f1d6gn .stVerticalBlock {
        gap: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)




if "selected" not in st.session_state:
    st.session_state.selected = "Home"

if "selected" in st.session_state:
    st.session_state.selected = st.session_state.selected

if st.session_state.get('switch_button', False):
    st.session_state['menu_option'] = 1
    manual_select = st.session_state['menu_option']
else:
    manual_select = None

if "page_number" not in st.session_state:
    st.session_state.page_number = 1

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

anatomical_label_dict = {
     'head': 'Head',
     'neck': 'Neck',
     'thorax': 'Thorax',
     'abdomen': 'Abdomen',
     'pelvis': 'Pelvis',
     'upper_limb': 'Upper Limb',
     'lower_limb': 'Lower Limb'
}

@st.cache_data
def load_image_metadata(file_folder):
    df = pd.read_parquet(os.path.join(file_folder, 'image_metadata_website_version.parquet'))
    df.rename({'postprocessed_label_list': 'labels'}, axis = 1, inplace = True)
    return df

@st.cache_data
def load_cases(file_folder):
    df = pd.DataFrame()
    for file_ in ['cases_1_website_version.parquet', 'cases_2_website_version.parquet']:
      df = pd.concat([df, pd.read_parquet(os.path.join(file_folder, file_))], ignore_index=True)
    df = df.astype({'age': 'float', 'year': 'int', 'commercial_use_license': 'bool', 'gender': 'category'})
    return df

def apply_filters(cases_df, image_metadata_df, filter_dict, page_number, elements_per_page):

  # case_df filters
  case_conditions = True
  case_conditions &= (cases_df['year'] >= filter_dict['min_year'])
  case_conditions &= (cases_df['year'] <= filter_dict['max_year'])
  if filter_dict['license'] == 'commercial':
    case_conditions &= (cases_df['commercial_use_license'] == True)
  if filter_dict['gender'] != 'Any':
    case_conditions &= (cases_df['gender'] == filter_dict['gender'])
  if (filter_dict['min_age'] != 0) or (filter_dict['max_age'] != 100):
    case_conditions &= (cases_df['age'] >= filter_dict['min_age'])
    case_conditions &= (cases_df['age'] <= filter_dict['max_age'])

  # image filters
  image_conditions = pd.Series(True, index=image_metadata_df.index) # instead of True, to avoid error when no image filters are set.
  if filter_dict['image_type_label']:
    image_conditions &= (image_metadata_df.labels.str.contains(f"'{filter_dict['image_type_label']}'"))
  if filter_dict['anatomical_region_label']:
    image_conditions &= (image_metadata_df.labels.str.contains(f"'{filter_dict['anatomical_region_label']}'"))
  if filter_dict['caption_search']:
    image_conditions &= (image_metadata_df.caption.apply(lambda x: text_matches_conditions(x, filter_dict['caption_search'])))

  page_status = 'no_more_pages' # used for pagination, to know if there is a following page.

  if filter_dict['resource_type'] == 'text':
    filtered_df = pd.merge(cases_df[case_conditions][['case_id', 'case_text']], image_metadata_df[image_conditions][['case_id']].drop_duplicates(), on="case_id", how="inner")
    outcome_case_ids = []
    for index, row in filtered_df.iterrows():
      if text_matches_conditions(row['case_text'], filter_dict['case_search']):
        outcome_case_ids.append(row['case_id'])
      if len(outcome_case_ids) > (page_number * elements_per_page):
        outcome_case_ids = outcome_case_ids[-(elements_per_page+1):-1]
        page_status = 'more_pages_left'
        break
    return (outcome_case_ids[-elements_per_page:], page_status)

  elif filter_dict['resource_type'] == 'image':
    filtered_df = pd.merge(cases_df[case_conditions][['case_id', 'case_text']], image_metadata_df[image_conditions][['case_id', 'file', 'caption']], on="case_id", how="inner")
    ids = []
    for index, row in filtered_df.iterrows():
      if text_matches_conditions(row['case_text'], filter_dict['case_search']):
        ids.append({'file': row['file'], 'case_id': row['case_id'], 'caption': row['caption']})
      if len(ids) > (page_number * elements_per_page):
        ids = ids[-(elements_per_page+1):-1]
        page_status = 'more_pages_left'
        break
    return (ids[-elements_per_page:], page_status)

  elif filter_dict['resource_type'] == 'both':
    filtered_df = pd.merge(cases_df[case_conditions][['case_id', 'case_text']], image_metadata_df[image_conditions][['case_id', 'file', 'caption']], on="case_id", how="inner").groupby('case_id').agg({'case_text': 'first', 'file': list, 'caption': list}).reset_index()
    outcome = []
    for index, row in filtered_df.iterrows():
      if text_matches_conditions(row['case_text'], filter_dict['case_search']):
        outcome.append({'case_id': row['case_id'], 'case_text': row['case_text'], 'images': dict(zip(row['file'], row['caption']))})
      if len(outcome) > (page_number * elements_per_page):
        outcome = outcome[-(elements_per_page+1):-1]
        page_status = 'more_pages_left'
        break
    if filter_dict['resource_type'] == 'text':
      return (outcome[-elements_per_page:], page_status)
    elif filter_dict['resource_type'] == 'both':
      return (outcome[-elements_per_page:], page_status)

def text_matches_conditions(text, query):
    """
    Checks if a given text meets all conditions defined in the parsed list.
    """
    parsed_list = parse_search_string(query)
    text = text.lower()  # Make text lowercase for case-insensitivity
    for condition in parsed_list:
        operator = condition['operator']
        substrings = condition['substring']
        if operator == "AND":
            # True if the text contains any of the substrings as full words
            if not any(full_word_match(text, sub) for sub in substrings):
                return False
        elif operator == "NOT":
            # False if the text contains any of the substrings as full words
            if any(full_word_match(text, sub) for sub in substrings):
                return False
    return True


def parse_search_string(query):
    """
    Parses a search string into a list of dictionaries with operators and substrings.
    """
    # Split the query by AND and NOT while keeping the delimiters
    tokens = re.split(r'(?<=\b)(AND|NOT)(?=\b)', query, flags=re.IGNORECASE)
    parsed_list = []
    current_operator = "AND"  # Default operator

    for token in map(str.strip, tokens):
        if token.upper() in {"AND", "NOT"}:
            current_operator = token.upper()
        elif token:
            # Extract substrings split by OR, clean leading/trailing non-alphanumeric characters
            substrings = [
                re.sub(r'^[^\w\s]+|[^\w\s]+$', '', sub.strip(' "').strip())
                for sub in token.lower().split(" or ")
            ]
            parsed_list.append({'operator': current_operator, 'substring': substrings})

    return parsed_list

def full_word_match(text, word):
    """
    Checks if a word matches as a full word in the text, ignoring case and boundaries.
    """
    # Use regex with word boundaries for full-word match
    return re.search(rf'\b{re.escape(word.lower())}\b', text) is not None


# ---------- STREAMLIT CODE --------------

with st.sidebar:
    st.logo("multicare-logo2.webp", size="large")
    selected = option_menu(
        menu_title=None,
        options=["Home", "Search", "About"],
        icons=["house", "search", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        manual_select=manual_select,
        key="selected",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "nav-link-selected": {"background-color": "#12588ECC", "font-weight": 700},
        },
    )
    st.header("Resource Usage")
    st.write(f"Memory Usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")


if selected == "Home":
    st.title("The Clinical Case Hub")
    st.markdown(
        f"""
        Welcome to The Clinical Case Hub, a platform designed to empower healthcare professionals and medical 
        students with real-world clinical cases. Our mission is to provide you with a diverse collection of 
        cases and images sourced from PubMed Central case reports, enabling you to enhance your diagnostic, 
        clinical decision-making, and critical thinking skills.
        """
    )
    st.image('clinical-hub.webp')
    st.button(f"Start your search‎ ‎ ‎→", key='switch_button')

        
elif selected == "Search":
    st.title("The Clinical Case Hub")
    st.write(
        """
        Refine your search with filters to find the clinical cases that align with your research focus or 
        learning goals. You can filter by different criteria, such as age, gender or content of the clinical 
        case text. Select a resource type based on your interests—text, image, or both.
        """
    )
    with st.form("filter_form"):
        st.subheader("Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_year, max_year = st.slider("Year", 1990, 2024, (2014, 2024))
            resource = st.selectbox("Resource Type", options=['text', 'image', 'both'], index=0)
            image_type_label = st.selectbox("Image Type Label", options=[None] + list(label_dict.values()))
            if image_type_label is not None:
                image_type_label = next((key for key, value in label_dict.items() if value == image_type_label), None)
        
        with col2:
            min_age, max_age = st.slider("Age", 0, 100, (18, 65))
            gender = st.selectbox("Gender", options=['Any', 'Female', 'Male'], index=0)
            anatomical_region_label = st.selectbox(
                "Anatomical Region Label", 
                options=[None] + list(anatomical_label_dict.values()),
                help="This filter can only be combined with specific image types: 'CT scan,' 'MRI,' 'X-ray,' 'Ultrasound,' 'Angiography,' and 'Nuclear Medicine'."
            )
            if anatomical_region_label is not None:
                anatomical_region_label = next((key for key, value in anatomical_label_dict.items() if value == anatomical_region_label), None)

        with col3:
            license = st.radio("License", options=['all', 'commercial'], horizontal=True, index=0)
            case_search = st.text_input(
                "Case Text Search", 
                value='', 
                help="Search operators: 'OR', 'AND', 'NOT'. Groups of terms that refer to the same concept should be concatenated using 'OR'. Use 'AND' to include a new term or group of terms, and use 'NOT' to exclude them. For example: '(CT OR tomography) AND (chest OR thorax) NOT abdomen' will return chest CT scans with no mentions of the word 'abdomen'."
            )
            caption_search = st.text_input(
                "Caption Text Search", value='', 
                help="Search operators: 'OR', 'AND', 'NOT'. Groups of terms that refer to the same concept should be concatenated using 'OR'. Use 'AND' to include a new term or group of terms, and use 'NOT' to exclude them. For example: '(CT OR tomography) AND (chest OR thorax) NOT abdomen' will return chest CT scans with no mentions of the word 'abdomen'."
            )
            
        submitted = st.form_submit_button("Search")

    if submitted: 
        st.markdown("####")
        st.subheader("Seach Results")
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
            'license': license,
            'resource_type': resource
        }

        # Load data
        image_metadata_df = load_image_metadata('.')
        cases_df = load_cases('.')

        if not image_metadata_df.empty and not cases_df.empty:
            page_number = st.session_state.page_number
            elements_per_page = 10
            st.session_state.filter_dict = filter_dict
    
    
        if filter_dict['resource_type'] == 'text':
            outcome, page_status = apply_filters(cases_df, image_metadata_df, filter_dict, page_number, elements_per_page)
            for case_id in outcome:
                row = cases_df[cases_df.case_id == case_id].iloc[0]      
                with st.expander(f"**{row['title']}** \n\n _Case ID:_ **{row['case_id']}** _Gender:_ **{row['gender']}** _Age:_ **{int(row['age'])}**"):
                    st.write(f"{row['case_text']}")
                    st.divider()
                    st.write(f"**Source**: _{row['citation']}_")   
              
            col1, col2, col3 = st.columns([1, 5, 1])
            with col1:
                if st.session_state.page_number > 1:
                    if st.button("⏮  Previous"):
                        st.session_state.page_number -= 1
            with col3:
                if page_status == "more_pages_left":
                    if st.button("Next  ⏭"):
                        st.session_state.page_number += 1
                        page_number = st.session_state.page_number
                        outcome, page_status = apply_filters(cases_df, image_metadata_df, filter_dict, page_number, elements_per_page)
                        for case_id in outcome:
                            row = cases_df[cases_df.case_id == case_id].iloc[0]      
                            with st.expander(f"**{row['title']}** \n\n _Case ID:_ **{row['case_id']}** _Gender:_ **{row['gender']}** _Age:_ **{int(row['age'])}**"):
                                st.write(f"{row['case_text']}")
                                st.divider()
                                st.write(f"**Source**: _{row['citation']}_")    


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
    col1, col2 = st.columns([3,2])
    with col1:
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
    with col2:
        st.image('medical_doctor_desktop.webp')


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
            <div style='text-align: center;'>
                <img src='data:image/jpeg;base64,{convert_image_to_base64(image_path)}' alt='{image_caption}' style="width: 35%; border-radius: 8px;'>
                <p><em>{image_caption}</em></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.write(f"Image Labels: {', '.join(image_labels)}")
        # st.write(f"Article Link: [Link]({article_link})")
        st.write(f"Citation: {article_citation}")

def convert_image_to_base64(image_path):
    from base64 import b64encode
    with open(image_path, "rb") as img_file:
        return b64encode(img_file.read()).decode("utf-8")

def display_case_both(cch, index):
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
                    <div style='text-align: center;'>
                        <img src='data:image/jpeg;base64,{convert_image_to_base64(image_path)}' alt='{image_caption}' style="width: 35%; border-radius: 8px;'>
                        <p><em>{image_caption}</em></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
        st.write(f"Image Labels: {', '.join(image_labels)}")
        # st.write(f"Article Link: [Link]({article_link})")
        st.write(f"Citation: {article_citation}")



