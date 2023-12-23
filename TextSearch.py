import streamlit as st
import pandas as pd
from streamlit_searchbox import st_searchbox

from google.cloud import discoveryengine_v1alpha
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct


# Constants for the API calls
PROJECT_ID = "721867696604"
DATA_STORE_ID = "villa-test-data-real_1702636375671"


@st.cache_resource
def get_client():
    return discoveryengine_v1alpha.SearchServiceClient()


def get_search_results(query, page_token=None):
    client = get_client()

    # Construct the request
    request = discoveryengine_v1alpha.SearchRequest(
        serving_config=f"projects/{PROJECT_ID}/locations/global/collections/default_collection/dataStores/{DATA_STORE_ID}/servingConfigs/default_search:search",
        query=query,
        page_token=page_token,
        query_expansion_spec=discoveryengine_v1alpha.SearchRequest.QueryExpansionSpec(
            condition=discoveryengine_v1alpha.SearchRequest.QueryExpansionSpec.Condition.AUTO
        ),
        spell_correction_spec=discoveryengine_v1alpha.SearchRequest.SpellCorrectionSpec(
            mode=discoveryengine_v1alpha.SearchRequest.SpellCorrectionSpec.Mode.AUTO
        ),
    )

    # Use the search method from the client, passing the request object
    response = client.search(request=request)
    return response


def get_all_search_results(query):
    all_results = []

    # Call the search method to get the pager
    pager = get_search_results(query)
    all_results.extend(pager.results)
    while pager.next_page_token:
        pager = get_search_results(query, pager.next_page_token)
        all_results.extend(pager.results)
    return all_results


@st.cache_data
def get_autocomplete_suggestions(searchterm: str):
    if not searchterm:
        return []

    client = discoveryengine_v1alpha.CompletionServiceClient()

    # Construct the request
    request = discoveryengine_v1alpha.CompleteQueryRequest(
        data_store=f"projects/{PROJECT_ID}/locations/global/collections/default_collection/dataStores/{DATA_STORE_ID}",
        query=searchterm,
        query_model="document-completable",
    )
    # Use the search method from the client, passing the request object
    response = client.complete_query(request=request)
    if hasattr(response, 'query_suggestions'):
        return [suggestion.suggestion for suggestion in response.query_suggestions]
    else:
        return []

def convert_struct_to_dict(struct_data):
    """Convert struct_data to a dictionary."""
    if struct_data:
        struct = Struct()
        struct.update(struct_data)
        return MessageToDict(struct)
    else:
        return {}


# Define the search function for st_searchbox


def main():
    st.title('Search Demo App')
    # print(get_autocomplete_suggestions("Sa"))
    # st.write(get_autocomplete_suggestions("Sa"))

    # # Autocomplete search box
    query = st_searchbox(
        search_function=get_autocomplete_suggestions,
        placeholder="Search ...",
        label="Search",
        key="searchboxx", 
        clear_on_submit=True,
        rerun_on_update=True  # This will trigger a rerun of the app on user input
    )

    # # st.write(query)
    # # st.write(get_search_results(query))
    if query:
        # Retrieve all results
        all_results = get_all_search_results(query)

        # Adjusted match condition to check for exact match disregarding capitalization
        match_results = [
            res
            for res in all_results
            if res.document.struct_data.get("content_en", "").lower().strip()== query.lower().strip()
            or res.document.struct_data.get("content_th", "").strip() == query.strip()
        ]
        non_match_results = [res for res in all_results if res not in match_results]

        # Sorting results
        match_results.sort(
            key=lambda x: (
                x.document.struct_data.get("villa_category_l3_en", ""),
                x.document.struct_data.get("villa_category_l2_en", ""),
            )
        )
        non_match_results.sort(
            key=lambda x: (
                x.document.struct_data.get("villa_category_l3_en", ""),
                x.document.struct_data.get("villa_category_l2_en", ""),
            )
        )

        # Concatenate match and non-match results
        combined_results = match_results + non_match_results
        # st.write(match_results)

        # Pagination setup
        page_size = 10
        total_pages = (len(combined_results) + page_size - 1) // page_size
        if "page_number" not in st.session_state:
            st.session_state.page_number = 1

        offset = (st.session_state.page_number - 1) * page_size
        paginated_results = combined_results[offset : offset + page_size]

        # Displaying results
       
        if paginated_results:
            df = pd.DataFrame(
                [
                    convert_struct_to_dict(item.document.struct_data)
                    for item in paginated_results
                ]
            )
            # Set a new index for the DataFrame that reflects the overall position in results
            df.index = range(1 + offset, 1 + offset + len(df))
            # Do not reset the index, so it does not become a column
            # Rename the index to display as 'Index' in the table
            df.index.name = "Index"
            columns_order = [
                "content_en",
                "content_th",
                "villa_category_l3_en",
                "villa_category_l2_en",
            ] + [
                col
                for col in df.columns
                if col
                not in [
                    "content_en",
                    "content_th",
                    "villa_category_l3_en",
                    "villa_category_l2_en",
                ]
            ]
            df = df[columns_order]
            st.table(df)  # Display DataFrame as a table without resetting the index
            st.write(
                f'Showing results {offset+1}-{min(offset+page_size, len(combined_results))} out of {len(combined_results)} for "{query}"'
            )

        # Navigation buttons below the table
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("First"):
                st.session_state.page_number = 1
        with col2:
            if st.button("Previous"):
                if st.session_state.page_number > 1:
                    st.session_state.page_number -= 1
        with col3:
            if st.button("Next"):
                if st.session_state.page_number < total_pages:
                    st.session_state.page_number += 1
        with col4:
            if st.button("Last"):
                st.session_state.page_number = total_pages

if __name__ == "__main__":
    main()
