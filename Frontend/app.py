import os
import streamlit as st
import requests

API_URL = os.getenv('API_URL', 'http://localhost:8000')

st.set_page_config(page_title='RAG Document Q&A', page_icon='📄', layout='centered')
st.title('📄 RAG Document Q&A')
st.caption('Upload a document and ask questions about it — with confidence scoring and knowledge gap detection')

# ── File Upload ──────────────────────────────────────────────────────────────

st.subheader('Step 1: Upload a Document')
uploaded = st.file_uploader('Choose a PDF, DOCX, or TXT file', type=['pdf', 'docx', 'txt'])

if uploaded:
    if st.button('Ingest Document'):
        with st.spinner(f'Ingesting {uploaded.name}...'):
            response = requests.post(
                f'{API_URL}/upload',
                files={'file': (uploaded.name, uploaded.getvalue())}
            )
        if response.status_code == 200:
            st.success(f'Document ingested successfully: {uploaded.name}')
            st.session_state['document_ready'] = True
            st.session_state['document_name'] = uploaded.name
        else:
            st.error(f'Upload failed: {response.json().get("detail", "Unknown error")}')

# Show which document is currently loaded
if st.session_state.get('document_ready'):
    st.info(f'Active document: {st.session_state["document_name"]}')

st.divider()

# ── Question & Answer ─────────────────────────────────────────────────────────

st.subheader('Step 2: Ask a Question')
question = st.text_input('Type your question here', placeholder='e.g. What methodology was used in this paper?')

if st.button('Ask', disabled=not question):
    if not st.session_state.get('document_ready'):
        st.warning('Please upload and ingest a document first.')
    else:
        with st.spinner('Searching document and generating answer...'):
            response = requests.post(
                f'{API_URL}/ask',
                json={'question': question, 'collection': 'docs'}
            )

        if response.status_code != 200:
            st.error(f'Error: {response.json().get("detail", "Unknown error")}')
        else:
            data = response.json()
            conf = data['confidence']

            # ── Confidence Badge ──────────────────────────────────────────────
            colour = {'high': 'green', 'medium': 'orange', 'low': 'red'}[conf['level']]
            st.markdown(
                f'**Confidence:** :{colour}[{conf["level"].upper()}] &nbsp;|&nbsp; '
                f'Score: `{conf["score"]}`'
            )

            # ── Knowledge Gap Warning ─────────────────────────────────────────
            if conf['is_knowledge_gap']:
                st.warning(f'Knowledge Gap Detected: {conf["gap_reason"]}')

            # ── Answer ────────────────────────────────────────────────────────
            st.markdown('### Answer')
            st.write(data['answer'])

            # ── Sources ───────────────────────────────────────────────────────
            with st.expander('View Sources Used'):
                for i, source in enumerate(data['sources'], 1):
                    st.markdown(
                        f'**Source {i}:** `{source["file"]}` — '
                        f'Page {source["page"]} — '
                        f'Relevance: `{source["score"]}`'
                    )
