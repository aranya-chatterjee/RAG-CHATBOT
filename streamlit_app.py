import streamlit as st
import requests

API_URL = "http://localhost:8000"

def upload_document():
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
    if uploaded_file:
        resp = requests.post(
            f"{API_URL}/documents/",
            files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        )
        if resp.status_code == 200:
            st.success("Document uploaded!")
        else:
            st.error(f"Upload failed: {resp.text}")

def list_documents():
    resp = requests.get(f"{API_URL}/documents/")
    if resp.status_code == 200:
        docs = resp.json()
        doc_names = [f"{d['id']}: {d['filename']}" for d in docs]
        doc_ids = [d['id'] for d in docs]
        return dict(zip(doc_names, doc_ids))
    else:
        st.error("Could not fetch documents list")
        return {}

def show_chat(doc_id):
    st.subheader("Chat History")
    resp = requests.get(f"{API_URL}/chat/{doc_id}/")
    chats = resp.json() if resp.status_code == 200 else []
    for chat in chats:
        st.markdown(f"**You:** {chat['user_message']}")
        st.markdown(f"**Masterji:** {chat['assistant_message']}\n---")
    
    st.subheader("Ask a question")
    question = st.text_input("Your question:")
    if st.button("Ask", key=f"ask_{doc_id}") and question:
        resp = requests.post(
            f"{API_URL}/chat/{doc_id}/",
            json={"message": question}
        )
        if resp.status_code == 200:
            answer = resp.json()["assistant_message"]
            st.success(f"Masterji: {answer}")
        else:
            st.error("Failed to get a response")

def main():
    st.title("RAG-Chatbot (Streamlit Frontend)")
    st.sidebar.header("Document Actions")
    upload_document()

    st.sidebar.header("Select Document")
    docs = list_documents()
    if docs:
        sel = st.sidebar.radio("Choose document", list(docs.keys()))
        doc_id = docs[sel]
        show_chat(doc_id)
    else:
        st.info("No documents yet. Please upload one.")

if __name__ == "__main__":
    main()