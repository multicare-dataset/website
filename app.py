import streamlit as st
import pandas as pd
import os
import re
from streamlit_option_menu import option_menu


# Streamlit page configuration
st.set_page_config(page_title="Clinical Case Hub", page_icon=":stethoscope:", layout="wide")


if 'search_executed' not in st.session_state:
    st.session_state['search_executed'] = False

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


if 'filter_dict' not in st.session_state:
    st.session_state['filter_dict'] = {
        'min_age': 18,
        'max_age': 65,
        'gender': 'Any',
        'case_search': '',
        'image_type_label': None,
        'anatomical_region_label': None,
        'caption_search': '',
        'min_year': 2014,
        'max_year': 2024,
        'license': 'all',
        'resource_type': 'text',
    }

if "filter_dict" in st.session_state:
    st.session_state.filter_dict = st.session_state.filter_dict

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


team_members = [
    {
        "name": "Mauro Nievas Offidani",
        "title": "MD, MSc, Data Scientist",
        "linkedin": "https://www.linkedin.com/in/mauronievasoffidani/",
        "image": "team/mauro-nievas-offidani.png",
    },
    {
        "name": "María Carolina González Galtier",
        "title": "MD, MA, Data Analyst",
        "linkedin": "https://www.linkedin.com/in/carogaltier/",
        "image": "team/carolina-gonzalez-galtier.png",
    },
    {
        "name": "Miguel Massiris",
        "title": "Role Description",
        "linkedin": "#",
        "image": "team/team-user.png",
    },
    {
        "name": "Facundo Roffet",
        "title": "Role Description",
        "linkedin": "#",
        "image": "team/team-user.png",
    },
    {
        "name": "Claudio Delrieux",
        "title": "Role Description",
        "linkedin": "#",
        "image": "team/team-user.png",
    },
]

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

def highlight_text(text, query, highlight_class='case-highlight'):
    """
    Devuelve el texto con las palabras de la query resaltadas.
    Se resaltan solo las condiciones AND (o por defecto), no las NOT.
    """
    parsed_list = parse_search_string(query)
    # Para simplificar, se resalta todo lo que no esté bajo NOT.
    # Las condiciones "AND" las tratamos resaltando cada término.
    # Cada condición puede tener varios términos separados por OR.
    # Se resaltarán todos esos términos.
    
    for condition in parsed_list:
        if condition['operator'] == "AND":
            for sub in condition['substring']:
                # Resaltar usando regex full-word, case-insensitive
                pattern = rf'(?i)\b({re.escape(sub)})\b'
                replacement = f"<mark class='{highlight_class}'>\\1</mark>"
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        # NOT: No se resaltan, simplemente no se hace nada.
    return text



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
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image('medical_doctor_desktop.webp')

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
            min_year, max_year = st.slider(
                "Year", 
                1990, 
                2024, 
                (st.session_state['filter_dict']['min_year'], st.session_state['filter_dict']['max_year'])
            )
            resource = st.selectbox(
                "Resource Type", 
                options=['text', 'image', 'both'], 
                index=['text', 'image', 'both'].index(st.session_state['filter_dict']['resource_type'])
            )
            image_type_index = ([None] + list(label_dict.values())).index(st.session_state['filter_dict']['image_type_label']) if st.session_state['filter_dict']['image_type_label'] in label_dict.values() else 0
            image_type_label = st.selectbox(
                "Image Type Label", 
                options=[None] + list(label_dict.values()),
                index=image_type_index
            )
            if image_type_label is not None:
                image_type_label = next((key for key, value in label_dict.items() if value == image_type_label), None)
        
        with col2:
            min_age, max_age = st.slider(
                "Age", 
                0, 
                100, 
                (st.session_state['filter_dict']['min_age'], st.session_state['filter_dict']['max_age'])
            )
            gender = st.selectbox(
                "Gender", 
                options=['Any', 'Female', 'Male'], 
                index=['Any', 'Female', 'Male'].index(st.session_state['filter_dict']['gender'])
            )
            anatomical_region_index = (
                [None] + list(anatomical_label_dict.values())
            ).index(st.session_state['filter_dict']['anatomical_region_label']) \
                if st.session_state['filter_dict']['anatomical_region_label'] in anatomical_label_dict.values() else 0
            anatomical_region_label = st.selectbox(
                "Anatomical Region Label", 
                options=[None] + list(anatomical_label_dict.values()),
                index=anatomical_region_index,
                help="This filter can only be combined with specific image types: 'CT scan,' 'MRI,' 'X-ray,' 'Ultrasound,' 'Angiography,' and 'Nuclear Medicine'."
            )
            if anatomical_region_label is not None:
                anatomical_region_label = next((key for key, value in anatomical_label_dict.items() if value == anatomical_region_label), None)

        with col3:
            license = st.radio(
                "License", 
                options=['all', 'commercial'], 
                horizontal=True, 
                index=['all', 'commercial'].index(st.session_state['filter_dict']['license'])
            )
            case_search = st.text_input(
                "Case Text Search", 
                value=st.session_state['filter_dict']['case_search'],
                help="Search operators: 'OR', 'AND', 'NOT'. Groups of terms that refer to the same concept should be concatenated using 'OR'. Use 'AND' to include a new term or group of terms, and use 'NOT' to exclude them. For example: '(CT OR tomography) AND (chest OR thorax) NOT abdomen' will return chest CT scans with no mentions of the word 'abdomen'."
            )
            caption_search = st.text_input(
                "Caption Text Search", 
                value=st.session_state['filter_dict']['caption_search'],
                help="Search operators: 'OR', 'AND', 'NOT'. Groups of terms that refer to the same concept should be concatenated using 'OR'. Use 'AND' to include a new term or group of terms, and use 'NOT' to exclude them. For example: '(CT OR tomography) AND (chest OR thorax) NOT abdomen' will return chest CT scans with no mentions of the word 'abdomen'."
            )
            
        submitted = st.form_submit_button("Search")

    image_metadata_df = load_image_metadata('.')
    cases_df = load_cases('.')
    if submitted: 
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
        if filter_dict != st.session_state.filter_dict:
            st.session_state.filter_dict = filter_dict
        st.session_state.search_executed = True
        st.session_state.page_number = 1
        elements_per_page = 10
        
    if "filter_dict" in st.session_state and st.session_state.search_executed: 
        st.subheader("Seach Results")
        page_number = st.session_state.page_number
        elements_per_page = 10
        
        outcome, page_status = apply_filters(
            cases_df, 
            image_metadata_df, 
            st.session_state.filter_dict, 
            page_number, 
            elements_per_page
        )


        top_col1, top_col2, top_col3 = st.columns([1, 3, 1])
        with top_col1:
            if page_number > 1:
                if st.button(" ⏮ Previous "):
                    st.session_state.page_number = page_number - 1
                    st.rerun()
    
        with top_col3:
            if page_status == "more_pages_left":
                if st.button(" Next ⏭ "):
                    st.session_state.page_number = page_number + 1
                    st.rerun()
        
        if outcome:
            if st.session_state.filter_dict['resource_type'] == 'text':
                for case_id in outcome:
                    row = cases_df[cases_df.case_id == case_id].iloc[0]   
                    age = int(row['age']) if not pd.isna(row['age']) else "Unknown"
                    with st.expander(f"**{row['title']}** \n\n **_Case ID:_ {row['case_id']}** **_Gender:_ {row['gender']}** **_Age:_ {age}**"):
                        st.divider()
                        st.markdown("#### Case Description", unsafe_allow_html=True)
                        highlighted_text = highlight_text(row['case_text'], st.session_state.filter_dict['case_search'], highlight_class='case-highlight')
                        st.markdown(highlighted_text, unsafe_allow_html=True)
                        #st.write(f"{row['case_text']}")
                        st.divider()
                        st.write(f"**Source**: _{row['citation']}_")

            elif st.session_state.filter_dict['resource_type'] == 'both':
                for case_ in outcome:
                    row = cases_df[cases_df.case_id == case_['case_id']].iloc[0]
                    age = int(row['age']) if not pd.isna(row['age']) else "Unknown"
                    with st.expander(f"**{row['title']}** \n\n **_Case ID:_ {row['case_id']}** **_Gender:_ {row['gender']}** **_Age:_ {age}**"):
                        st.divider()
                        st.markdown("#### Case Description")
                        st.write(f"{row['case_text']}")
                        st.markdown("#### Images")
                        
                        images_list = list(case_['images'].items())
                        for i in range(0, len(images_list), 2):
                            pair = images_list[i:i+2]
                            cols = st.columns(2)
                            for idx, (file_name, caption) in enumerate(pair):
                                with cols[idx]:
                                    st.image(f"img/{file_name}", caption=caption)
                            
                        st.divider()
                        st.write(f"**Source**: _{row['citation']}_")
            
            elif st.session_state.filter_dict['resource_type'] == 'image':
                for i in range(0, len(outcome), 2):
                    pair = outcome[i:i+2]
                    cols = st.columns(2)  # Crear dos columnas
                    for idx, image_dict in enumerate(pair):
                        row = cases_df[cases_df.case_id == image_dict['case_id']].iloc[0]
                        age = int(row['age']) if not pd.isna(row['age']) else "Unknown"
                        with cols[idx]: 
                            with st.container(border=True):
                                st.image(f"img/{image_dict['file']}", caption=image_dict['caption'])
                                st.write(f"_Case ID:_ **{row['case_id']}** | _Gender:_ **{row['gender']}**  | _Age:_ **{age}**")

                                st.write(f"**Source**: _{row['citation']}_")
    
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                if page_number > 1:
                    if st.button("⏮ Previous"):
                        st.session_state.page_number = page_number - 1
                        st.rerun()
    
            with col3:
                if page_status == "more_pages_left":
                    if st.button("Next ⏭"):
                        st.session_state.page_number = page_number + 1
                        st.rerun()
        
        else:
            st.warning("No results found for the current filters.")
    

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
    columns = st.columns([1, 3, 1, 3, 1, 3, 1])
    
    for i, member in enumerate(team_members[:3]):
        with columns[i * 2 + 1]:
            st.image(member["image"])
            st.markdown(f"<p style='text-align: center; text-decoration: none; color: rgb(0, 104, 201);'><a href='{member['linkedin']}' target='_blank' style='text-decoration: none; color: inherit;'>{member['name']}</a></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: rgba(49, 51, 63, 0.6); font-size: 14px;'>{member['title']}</p>", unsafe_allow_html=True)
    
    # Segunda fila para los últimos dos miembros
    columns = st.columns([3, 3, 1, 3, 3])
    
    for i, member in enumerate(team_members[3:]):
        with columns[i * 2 + 1]:
            st.image(member["image"])
            st.markdown(f"<p style='text-align: center; text-decoration: none; color: rgb(0, 104, 201);'><a href='{member['linkedin']}' target='_blank' style='text-decoration: none; color: inherit;'>{member['name']}</a></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: rgba(49, 51, 63, 0.6); font-size: 14px;'>{member['title']}</p>", unsafe_allow_html=True)


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

    .stMainBlockContainer img {
        border-radius: 10px;
    }
    
    .stMainBlockContainer {
        padding-top: 3rem;
    }

    .stLogo {
        margin: 1rem auto;
        height: 28px;
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
        color: #fff !important;
    }
    
    .stFormSubmitButton button:hover, 
    .stFormSubmitButton button:active, 
    .stFormSubmitButton button:focus, 
    .stButton button:hover, 
    .stButton button:active, 
    .stButton button:focus, 
    .stDownloadButton button:hover, 
    .stDownloadButton button:active, 
    .stDownloadButton button:focus {
        padding: 0.5rem 1rem;
        background: rgba(18, 88, 142, 0.6);
        color: #fff !important;
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

    div[data-testid="stImageCaption"] {
        color: #000;
        font-size: 16px;
        
    }

    .stExpander details {
        padding-bottom: 3rem;
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
        width: 100%;
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

    details summary span [data-testid="stMarkdownContainer"] p:nth-of-type(2) em {
        font-weight: 400; 
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


    .case-highlight {
        background-color: yellow;
        color: black;
    }
    .caption-highlight {
        background-color: lightblue;
        color: black;
    }

    
    </style>
    """,
    unsafe_allow_html=True,
)
