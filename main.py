#Import necessary libraries
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
import streamlit as st

class QASystem:
	def __init__(self):
		# Load environment variables
		load_dotenv()

		# Initialize the large language model
		self.llm = GoogleGenerativeAI(
					model="gemini-1.5-flash",				# Gemini 1.5 Flash model
					google_api_key=os.environ["API_KEY"],	# API Key
					temperature=0.1							# Low temperature for deterministic response
					)

		# Initialize the embedding model
		self.huggingface_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

		# Vector database filepath
		self.vectordb_path = "./faiss_vectordb"

		# Dataset filepath
		self.dataset_path = "./dataset.csv"

		self.vectordb = None
		self.prompt = None
		self.chain = None

		# Download necessary NLTK data
		nltk.download('punkt')
		nltk.download('punkt_tab')
		nltk.download('stopwords')
		nltk.download('wordnet')

		# Initialize NLTK tools
		self.stop_words = set(stopwords.words('english'))
		self.lemmatizer = WordNetLemmatizer()

	'''
		Function to preprocess the dataset:
		- Case normalization, tokenization, stopword removal, lemmatization.
	'''
	def preprocess_data(self, text):
		# Case normalization
	    text = text.lower()

	    # Tokenization
	    tokens = word_tokenize(text)

	    # Stop word removal and lemmatization
	    processed_tokens = [
					self.lemmatizer.lemmatize(token)
					for token in tokens
					if token not in self.stop_words
				]

	    # Join the processed tokens into a string
	    processed_data = ' '.join(processed_tokens)

	    return processed_data

	'''
	    Function to create a vector database:
	    - Dataset is loaded.
	    - Dataset is converted into vectors using the embedding model.
	    - FAISS vector database is created.
	    - Database is saved locally for later use.
	'''
	def create_vectordb(self):
		# Load CSV file
		loader = CSVLoader(file_path=self.dataset_path, source_column="questions")
		data = loader.load()

		# Preprocess the documents
		for doc in data:
			doc.page_content = self.preprocess_data(doc.page_content)

		# Create vector database
		self.vectordb = FAISS.from_documents(documents=data, embedding=self.huggingface_embedding)

		# Save database locally
		self.vectordb.save_local(self.vectordb_path)

	'''
    	Function to load the vector database.
    '''
	def load_vectordb(self):
		# Load vector database
		self.vectordb = FAISS.load_local(
					            self.vectordb_path,
					            embeddings=self.huggingface_embedding,
					            allow_dangerous_deserialization=True
					        	)

	'''
    	Function to create prompt for the LLM.
    '''
	def create_prompt(self):
		# Define the prompt template
		prompt_template = """Use the following context to answer the question briefly. \
        If you don't know the answer, just say "I don't know.🤷‍♂️ Do you want to notify this to the TA?". \
        Don't try to make up an answer.

        CONTEXT: {context}
        QUESTION: {question}"""

		# Create the prompt
		self.prompt = PromptTemplate(
						template=prompt_template,
						input_variables=["context", "question"]
						)
		return self.prompt

	'''
    	Function to create question-answer chain.
    '''
	def create_chain(self, prompt):
		# Create a retriever from the vector database
		retriever = self.vectordb.as_retriever()

		# Create QA chain
		self.chain = RetrievalQA.from_chain_type(
								        llm=self.llm,
								        retriever=retriever,
								        chain_type_kwargs={"prompt": self.prompt},
								        return_source_documents=True
								    	)
		return self.chain

	'''
    	Function to run Streamlit Web Application UI
    '''
	def run(self):
		# Page title and caption
		st.title("AI Teaching Assistant 🤓")
		st.caption("Try asking me: Where can I find the course syllabus? When is the final exam? Provide some reference books, etc.")
		st.subheader("Question:")

		# User's question
		user_question = st.text_input(
							label="user_question",
							placeholder="Type your question here...",
							label_visibility="collapsed"
							)
		
		if user_question:
			prompt = self.create_prompt()
			qa_chain = self.create_chain(prompt)
			response = qa_chain(user_question)

			# Display the answer
			st.subheader("Answer:")
			st.write(response["result"])

			# For internal
			print(f"User's question: {response["query"]}")
			print(f"LLM's response: {response["result"]}")

			# Fetch top 3 relevant documents with similarity scores for the user question
			relevant_docs = self.vectordb.similarity_search_with_relevance_scores(user_question, k=3)

			print("Relevant documents")
			print("---")
			for rdocs, score in relevant_docs:
				print(f"> {rdocs.page_content} [Similarity={score:.2f}]")


			if "I don't know" in response["result"]:
				btn = st.link_button(
							label="Notify TA",
							url="mailto:ta@northeastern.edu",
							help="ta@northeastern.edu",
							icon="✉️"
							)


def main():
	# Create an instance
	qa_system = QASystem()

	# Create vector database if not already
	if not qa_system.vectordb:
		qa_system.create_vectordb()

	# Load the database
	qa_system.load_vectordb()

	# Run application
	qa_system.run()


if __name__ == "__main__":
	main()

