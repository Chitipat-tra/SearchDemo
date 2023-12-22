import streamlit as st
import requests
import pandas as pd
import json
import base64
from PIL import Image
from io import BytesIO
from google.oauth2 import service_account
import google.auth.transport.requests
from google.cloud import discoveryengine_v1beta

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part


# Constants for the API calls
PROJECT_ID = "721867696604"
DATA_STORE_ID = "villa-test-data-real_1702636375671"
AUTOCOMPLETE_MODEL = 'document-completable' # Securely manage and store the access token
SEARCH_ENDPOINT = f"https://discoveryengine.googleapis.com/v1alpha/projects/{PROJECT_ID}/locations/global/collections/default_collection/dataStores/{DATA_STORE_ID}/servingConfigs/default_search:search"
AUTOCOMPLETE_ENDPOINT = f"https://discoveryengine.googleapis.com/v1beta/projects/{PROJECT_ID}/locations/global/collections/default_collection/dataStores/{DATA_STORE_ID}:completeQuery"


# Function to create credentials
def get_credentials():
    credentials, project = google.auth.default()
    return credentials


@st.cache_data
def get_search_results(query, offset=0, page_size=50):
    credentials = get_credentials()
    authed_session = google.auth.transport.requests.AuthorizedSession(credentials)

    data = {
        "query": query,
        "pageSize": page_size,
        "offset": offset,
        "queryExpansionSpec": {"condition": "AUTO"},
        "spellCorrectionSpec": {"mode": "AUTO"}
    }

    response = authed_session.post(SEARCH_ENDPOINT, json=data)
    return response.json()


def get_all_search_results(query):
    all_results = []
    offset = 0
    page_size = 100  # API's maximum limit per call

    # Initial API call to get the total number of results
    initial_response = get_search_results(query, offset, page_size)
    total_results = initial_response.get('totalSize', 0)
    # st.write(total_results)

    all_results.extend(initial_response.get('results', []))
    # st.write(all_results)
    # Fetch remaining results in batches of 50
    while len(all_results) < total_results:
        offset += page_size
        response = get_search_results(query, offset, page_size)
        new_results = response.get('results', [])
        if not new_results:
            break  # No more results available
        all_results.extend(new_results)

    return all_results


def generate_text_from_image(image_data):
    vertexai.init(project=PROJECT_ID, location="asia-southeast1")
    multimodal_model = GenerativeModel("gemini-pro-vision")

    response = multimodal_model.generate_content(
        [
            "Analyze the image and identify the primary object relevant to supermarket items. Return a single, specific word that best describes this object for a search query. Format: '[single descriptive word]'.",
            Part.from_data(image_data, mime_type="image/jpeg")
        ]
    )
    return response.text


def main():
    st.title("Image Search Application")
    query=""
    uploaded_image = st.file_uploader("Upload an image for search", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        with BytesIO() as img_byte_arr:
            image.save(img_byte_arr, format='JPEG')
            image_data = img_byte_arr.getvalue()

        query = generate_text_from_image(image_data)
        st.write(f"Generated Query: {query}")

    if query:
        # Retrieve all results
        total_results = get_all_search_results(query)

        # Adjusted match condition to check for exact match disregarding capitalization
        # st.write( total_results['document']['structData'].get('content_th', ''))
        match_results = [res for res in total_results if res['document']['structData'].get('content_en', '').lower().strip() == query.lower().strip() or res['document']['structData'].get('content_th', '').strip() == query.strip()]

        non_match_results = [res for res in total_results if res not in match_results]
  
        # Sorting results
        match_results.sort(key=lambda x: (x['document']['structData'].get('villa_category_l3_en', ''), x['document']['structData'].get('villa_category_l2_en', '')))
        non_match_results.sort(key=lambda x: (x['document']['structData'].get('villa_category_l3_en', ''), x['document']['structData'].get('villa_category_l2_en', '')))

        # Concatenate match and non-match results
        combined_results = match_results + non_match_results
        # st.write(match_results)

        # Pagination setup
        page_size = 10
        total_pages = (len(combined_results) + page_size - 1) // page_size
        if 'page_number' not in st.session_state:
            st.session_state.page_number = 1

        offset = (st.session_state.page_number - 1) * page_size
        paginated_results = combined_results[offset:offset + page_size]

        # Displaying results
        # Displaying results with index
        if paginated_results:
            df = pd.DataFrame([item['document']['structData'] for item in paginated_results])
            # Set a new index for the DataFrame that reflects the overall position in results
            df.index = range(1 + offset, 1 + offset + len(df))
            # Do not reset the index, so it does not become a column
            # Rename the index to display as 'Index' in the table
            df.index.name = 'Index'
            columns_order = ['content_en', 'content_th', 'villa_category_l3_en', 'villa_category_l2_en'] + [col for col in df.columns if col not in ['content_en', 'content_th', 'villa_category_l3_en', 'villa_category_l2_en']]
            df = df[columns_order]
            st.table(df)  # Display DataFrame as a table without resetting the index
            st.write(f'Showing results {offset+1}-{min(offset+page_size, len(combined_results))} out of {len(combined_results)} for "{query}"')



        # Navigation buttons below the table
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button('First'):
                st.session_state.page_number = 1
        with col2:
            if st.button('Previous'):
                if st.session_state.page_number > 1:
                    st.session_state.page_number -= 1
        with col3:
            if st.button('Next'):
                if st.session_state.page_number < total_pages:
                    st.session_state.page_number += 1
        with col4:
            if st.button('Last'):
                st.session_state.page_number = total_pages


if __name__ == "__main__":
    main()
