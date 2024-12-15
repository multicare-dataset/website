import streamlit as st
import pandas as pd
import ast
import os
import re
from PIL import Image
from streamlit_option_menu import option_menu
import psutil

# Streamlit page configuration
st.set_page_config(page_title="Multicare Dataset", page_icon=":stethoscope:", layout="wide")

# Functions to load data with caching
@st.cache_data(ttl=3600, max_entries=5)
def load_article_metadata(file_folder):
    """Load article metadata from a parquet file."""
    df = pd.read_parquet(os.path.join(file_folder, 'article_metadata_website_version.parquet'))
    df['year'] = df['year'].astype(int)
    return df

@st.cache_data(ttl=3600, max_entries=5)
def load_image_metadata(file_folder):
    """Load image metadata from a parquet file."""
    df = pd.read_parquet(os.path.join(file_folder, 'image_metadata_website_version.parquet'))
    df.rename({'postprocessed_label_list': 'labels'}, axis=1, inplace=True)
    df['labels'] = df['labels'].apply(ast.literal_eval)
    return df

@st.cache_data(ttl=3600, max_entries=5)
def load_cases(file_folder, min_year, max_year):
    """Load case data from multiple parquet files based on the year range."""
    df = pd.DataFrame()
    for file_ in ['cases_1990_2012.parquet', 'cases_2013_2017.parquet', 'cases_2018_2021.parquet', 'cases_2022_2024.parquet']:
        years = file_.split('.')[0].split('_')[1:]
        if (max_year >= int(years[0])) and (min_year <= int(years[1])):
            df = pd.concat([df, pd.read_parquet(os.path.join(file_folder, file_))], ignore_index=True)
    return df

# Centralized profiling for resource usage
def display_resource_usage():
    st.write(f"Memory Usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
    st.write(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")

# Clinical Case Hub Class
class ClinicalCaseHub:
    def __init__(self, article_metadata_df, image_metadata_df, cases_df, image_folder='img'):
        self.image_folder = image_folder
        self.full_metadata_df = article_metadata_df.copy()
        self.full_image_metadata_df = image_metadata_df.copy()
        self.full_cases_df = cases_df.copy()
        self.full_cases_df['age'] = pd.to_numeric(self.full_cases_df['age'], errors='coerce', downcast='integer')

    def apply_filters(self, filter_dict):
        # Optimized filtering operations
        self.filter_dict = filter_dict

        self.metadata_df = self.full_metadata_df[
            (self.full_metadata_df.year >= self.filter_dict['min_year']) &
            (self.full_metadata_df.year <= self.filter_dict['max_year'])
        ].copy()
        if self.filter_dict['license'] == 'commercial':
            self.metadata_df = self.metadata_df[self.metadata_df.commercial_use_license]

        self.cases_df = self.full_cases_df[self.full_cases_df['article_id'].isin(self.metadata_df['article_id'])]
        if self.filter_dict['min_age']:
            self.cases_df = self.cases_df[self.cases_df.age >= self.filter_dict['min_age']]
        if self.filter_dict['max_age']:
            self.cases_df = self.cases_df[self.cases_df.age <= self.filter_dict['max_age']]
        if self.filter_dict['gender'] != 'Any':
            self.cases_df = self.cases_df[self.cases_df.gender == self.filter_dict['gender']]
        if self.filter_dict['case_search']:
            self.cases_df = self.cases_df[self.cases_df.case_text.str.contains(self.filter_dict['case_search'], case=False, na=False)]

        self.image_metadata_df = self.full_image_metadata_df[self.full_image_metadata_df['article_id'].isin(self.metadata_df['article_id'])]
        if self.filter_dict['image_type_label']:
            self.image_metadata_df = self.image_metadata_df[self.image_metadata_df.labels.apply(lambda x: self.filter_dict['image_type_label'] in x)]
        if self.filter_dict['anatomical_region_label']:
            self.image_metadata_df = self.image_metadata_df[self.image_metadata_df.labels.apply(lambda x: self.filter_dict['anatomical_region_label'] in x)]
        if filter_dict['caption_search']:
            self.image_metadata_df = self.image_metadata_df[self.image_metadata_df.caption.str.contains(filter_dict['caption_search'], case=False, na=False)]

        # Harmonizing filtered data
        self.filtered_article_ids = set(self.metadata_df['article_id']).intersection(
            self.cases_df['article_id'], self.image_metadata_df['article_id']
        )
        self.filtered_case_ids = set(self.cases_df['case_id']).intersection(self.image_metadata_df['case_id'])

        self.metadata_df = self.metadata_df[self.metadata_df['article_id'].isin(self.filtered_article_ids)]
        self.cases_df = self.cases_df[self.cases_df['case_id'].isin(self.filtered_case_ids)]
        self.image_metadata_df = self.image_metadata_df[self.image_metadata_df['case_id'].isin(self.filtered_case_ids)]

# ---------- STREAMLIT CODE --------------

def main():
    st.sidebar.title("Multicare Dataset")
    selected = option_menu(
        menu_title=None,
        options=["Search", "About"],
        icons=["search", "info-circle"],
        default_index=0
    )
    display_resource_usage()

    if selected == "Search":
        with st.form("filter_form"):
            st.subheader("Filters")
            col1, col2, col3 = st.columns(3)
            with col1:
                min_year, max_year = st.slider("Year", 1990, 2024, (2014, 2024))
            submitted = st.form_submit_button("Apply...")

if __name__ == "__main__":
    main()
