import streamlit as st
from rag_backend import generate_answer



st.set_page_config(page_title="BNS RAG Assistant")

st.title("⚖️ BNS Legal RAG Assistant")
st.markdown("Ask questions related to Bharatiya Nyaya Sanhita (BNS)")

query = st.text_area("Enter your legal question")

if st.button("Get Answer"):

    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing BNS provisions..."):
            answer, sources = generate_answer(query)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Sections")
        for i, src in enumerate(sources):
            st.markdown(f"**Source {i+1}**")
            st.text(src[:800])

