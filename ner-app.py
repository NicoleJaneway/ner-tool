import streamlit as st
import ktrain
import pickle

from displacy_utils import *

st.set_page_config(layout="wide")

header = st.beta_container()

text_input, text_output = st.beta_columns(2)

def get_predictions(text):
    predictor = ktrain.load_predictor('./predictor')
    return predictor.predict(text, return_proba=True)

with header:
    st.title("Atlas Research Named Entity Recognition Demo")

with text_input:
    st.header("Text Input")
    user_input = st.text_area("Add your clinical notes below (up to 250 characters):", value="", height=600)

with text_output:
    st.header("Tagged Output")
    if user_input != "":
        lst = get_predictions(user_input)
        text = " ".join([t[0] for t in lst])
        df = pd.DataFrame(lst)
        df.columns = ['token', 'IOBtag', 'prob']
        df = add_NER_tag(df)
        start_chars, end_chars = get_offset(lst)
        df['start'] = start_chars
        df['end'] = end_chars
        df = make_ent(df)
        ent_df = collapse(df)
        ent_lst = displacy_format(ent_df)
        output = {'text': text,
                  'ents': ent_lst}

        html = displacy.render(output, style="ent", manual=True, options=options)
        st.markdown(html, unsafe_allow_html=True)


about_section = st.beta_container()

with about_section:
    st.header("About")
    st.text("This application utilizes ClinicalBERT trained on data from the i2b2 2011 NLP challenge.")
