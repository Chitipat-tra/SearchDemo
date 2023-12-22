import streamlit as st
import requests
import pandas as pd
import json
from streamlit_searchbox import st_searchbox
from google.oauth2 import service_account
import google.auth.transport.requests


# Constants for the API calls
PROJECT_ID = "721867696604"
DATA_STORE_ID = "villa-test-data-real_1702636375671"
AUTOCOMPLETE_MODEL = 'document-completable' # Securely manage and store the access token
SEARCH_ENDPOINT = f"https://discoveryengine.googleapis.com/v1alpha/projects/{PROJECT_ID}/locations/global/collections/default_collection/dataStores/{DATA_STORE_ID}/servingConfigs/default_search:search"
AUTOCOMPLETE_ENDPOINT = f"https://discoveryengine.googleapis.com/v1beta/projects/{PROJECT_ID}/locations/global/collections/default_collection/dataStores/{DATA_STORE_ID}:completeQuery"


# Function to create credentials
def get_credentials():
    credentials = service_account.Credentials.from_service_account_file(
        './dvt-sg-vertex-ai-981ab18c1d48.json', scopes=["https://www.googleapis.com/auth/cloud-platform"])
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
    all_results.extend(initial_response.get('results', []))

    # Fetch remaining results in batches of 50
    while len(all_results) < total_results:
        offset += page_size
        response = get_search_results(query, offset, page_size)
        new_results = response.get('results', [])
        if not new_results:
            break  # No more results available
        all_results.extend(new_results)

    return all_results


@st.cache_data
def get_autocomplete_suggestions(searchterm: str):
    if not searchterm:
        return []

    # Create an authenticated session
    credentials = get_credentials()
    authed_session = google.auth.transport.requests.AuthorizedSession(credentials)

    params = {"query": searchterm, "query_model": AUTOCOMPLETE_MODEL}

    # Use the authenticated session to make the request
    response = authed_session.get(AUTOCOMPLETE_ENDPOINT, params=params)

    if response.status_code == 200:
        suggestions = response.json().get('querySuggestions', [])
        return [sugg['suggestion'] for sugg in suggestions]
    else:
        return []

# Define the search function for st_searchbox
def search_function(searchterm: str):
    return get_autocomplete_suggestions(searchterm)




def main():
    st.title('Search Demo App')

    # Autocomplete search box
    query = st_searchbox(
        search_function=search_function,
        placeholder="Search ...",
        label="Search",
        key="searchboxx", 
        clear_on_submit=True,
        rerun_on_update=True  # This will trigger a rerun of the app on user input
    )

    # st.write(query)
    # st.write(get_search_results(query))
    if query:
        # Retrieve all results
        total_results = get_all_search_results(query)

        # Adjusted match condition to check for exact match disregarding capitalization

        match_results = [res for res in total_results if res['document']['structData'].get('content_en', '').lower().strip() == query.lower().strip() or res['document']['structData'].get('content_th', '').strip() == query.strip()]

        non_match_results = [res for res in total_results if res not in match_results]
  
        # Sorting results
        match_results.sort(key=lambda x: (x['document']['structData'].get('villa_category_l3_en', ''), x['document']['structData'].get('villa_category_l2_en', '')))
        non_match_results.sort(key=lambda x: (x['document']['structData'].get('villa_category_l3_en', ''), x['document']['structData'].get('villa_category_l2_en', '')))

        # Concatenate match and non-match results
        combined_results = match_results + non_match_results


        # Pagination setup
        page_size = 10
        total_pages = (len(combined_results) + page_size - 1) // page_size
        if 'page_number' not in st.session_state:
            st.session_state.page_number = 1

        offset = (st.session_state.page_number - 1) * page_size
        paginated_results = combined_results[offset:offset + page_size]

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
