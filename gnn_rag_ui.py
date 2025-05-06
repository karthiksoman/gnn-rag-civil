import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize Pinecone client
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    st.sidebar.success("Pinecone connected!")
except Exception as e:
    st.sidebar.error(f"Pinecone connection error: {str(e)}")

# Initialize clients
try:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    st.sidebar.success("OpenAI connected!")
except Exception as e:
    st.sidebar.error(f"OpenAI connection error: {str(e)}")

try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    st.sidebar.success("Groq connected!")
except Exception as e:
    st.sidebar.error(f"Groq connection error: {str(e)}")

# Helper functions
def get_embedding(text, model="text-embedding-ada-002"):
    """Get embedding from OpenAI"""
    try:
        text = text.replace("\n", " ")
        return openai_client.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return None

def get_groq_response(instruction, system_prompt, chat_model, temperature=0.7):
    """Get response from Groq"""
    try:
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY is not set in the environment variables")
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction}
            ],
            model=chat_model,
            temperature=temperature
        )
        
        if chat_completion.choices:
            return chat_completion.choices[0].message.content
        else:
            return 'Unexpected response'
    except Exception as e:
        st.error(f"Error getting Groq response: {str(e)}")
        return f"Error: {str(e)}"

def semantic_retrieval_and_answer(query, top_k=10, model="llama3-8b-8192", temperature=0.7):
    """Perform semantic search and generate an answer"""
    try:
        with st.spinner("Performing semantic search..."):
            # Step 1: Get semantic embedding for the query
            query_embedding = get_embedding(query)
            
            if not query_embedding:
                return {
                    'semantic_results': [],
                    'answer': "Failed to generate embedding for query."
                }
            
            # Step 2: Search the semantic index
            semantic_index = pc.Index("rome-egypt-semantic")
            search_results = semantic_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Step 3: Format results for easier consumption
            semantic_results = []
            for match in search_results.matches:
                semantic_results.append({
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata
                })
            
            # Step 4: Format the retrieved context for the LLM
            context = ""
            for i, result in enumerate(semantic_results):
                metadata = result['metadata']
                context += f"Document {i+1} (Score: {result['score']:.4f}):\n"
                context += f"Source: {metadata.get('provenance', 'Unknown')}\n"
                if 'content' in metadata:
                    context += f"Content: {metadata['content']}\n\n"
                elif 'content_preview' in metadata:
                    context += f"Content Preview: {metadata['content_preview']}\n\n"
                else:
                    context += f"Base Document: {metadata.get('base_document', 'Unknown')}\n\n"
            
            # Step 5: Create the system prompt
            system_prompt = """You are a helpful assistant that provides accurate information based on the 
            search results provided. Answer the query using only the information in the retrieved documents.
            If the information needed to answer the query is not in the documents, acknowledge this limitation.
            Be concise and specific."""
            
            # Step 6: Create the instruction for the LLM
            instruction = f"""
            Query: {query}
            
            Context:
            {context}
            
            Based on the above context, please answer the query. 
            Provide specific details from the documents (which document, provenance etc) and indicate when information might be missing or uncertain.
            Do not give any score. Response should be in a conversational tone.
            Provide answer in elegant markdown format.
            """
            
            # Step 7: Get response from Groq
            answer = get_groq_response(instruction, system_prompt, model, temperature)
            
            # Return both results and answer
            return {
                'semantic_results': semantic_results,
                'answer': answer
            }
    
    except Exception as e:
        error_message = f"Error in semantic retrieval and answer generation: {str(e)}"
        st.error(error_message)
        return {
            'semantic_results': [],
            'answer': f"An error occurred: {error_message}"
        }

def graph_retrieval_and_answer(query, semantic_top_k=5, graph_neighbors_k=5, model="llama3-8b-8192", temperature=0.7):
    """Perform graph-enhanced search and generate an answer based on graph connections"""
    try:
        with st.spinner("Performing graph-enhanced search..."):
            # Step 1: First get semantic embedding for the query as entry points
            query_embedding = get_embedding(query)
            
            if not query_embedding:
                return {
                    'graph_results': [],
                    'answer': "Failed to generate embedding for query."
                }
            
            # Step 2: Get initial semantic results as entry points
            semantic_index = pc.Index("rome-egypt-semantic")
            semantic_results = semantic_index.query(
                vector=query_embedding,
                top_k=semantic_top_k,
                include_metadata=True
            )
            
            # Step 3: Access graph index
            graph_index = pc.Index("rome-egypt-graph")
            
            # Step 4: Find graph neighbors for each semantic result
            all_results = []
            seen_ids = set()
            
            # Add semantic entry points with a flag
            for match in semantic_results.matches:
                result = {
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata,
                    'is_entry_point': True  # Flag to indicate this is a direct match from semantic search
                }
                all_results.append(result)
                seen_ids.add(match.id)
            
            # For each semantic entry point, find graph neighbors
            for match in semantic_results.matches:
                try:
                    # Get the graph vector for this document ID
                    vector_response = graph_index.fetch(ids=[match.id])
                    
                    if match.id in vector_response.vectors:
                        # Get the graph embedding vector
                        graph_vector = vector_response.vectors[match.id].values
                        
                        # Find neighbors in graph space
                        neighbors = graph_index.query(
                            vector=graph_vector,
                            top_k=graph_neighbors_k + 1,  # +1 because the entry point itself will be included
                            include_metadata=True
                        )
                        
                        # Add neighbors to results if not already seen
                        for neighbor in neighbors.matches:
                            if neighbor.id not in seen_ids:
                                neighbor_result = {
                                    'id': neighbor.id,
                                    'score': neighbor.score,
                                    'metadata': neighbor.metadata,
                                    'is_entry_point': False,
                                    'connected_to': match.id  # Store which entry point this is connected to
                                }
                                all_results.append(neighbor_result)
                                seen_ids.add(neighbor.id)
                
                except Exception as e:
                    st.warning(f"Error fetching graph neighbors for {match.id}: {str(e)}")
            
            # Step 5: Format the context with graph structure highlighted
            context = "ENTRY POINTS (directly relevant to query):\n"
            entry_points = [r for r in all_results if r.get('is_entry_point', False)]
            
            for i, result in enumerate(entry_points):
                metadata = result['metadata']
                context += f"Entry Point {i+1} (Score: {result['score']:.4f}):\n"
                context += f"Source: {metadata.get('provenance', 'Unknown')}\n"
                
                if 'content' in metadata:
                    context += f"Content: {metadata['content']}\n\n"
                elif 'content_preview' in metadata:
                    context += f"Content Preview: {metadata['content_preview']}\n\n"
                else:
                    context += f"Base Document: {metadata.get('base_document', 'Unknown')}\n\n"
            
            # Add connected nodes
            context += "\nCONNECTED INFORMATION (related through knowledge graph):\n"
            neighbors = [r for r in all_results if not r.get('is_entry_point', False)]
            
            for i, result in enumerate(neighbors):
                metadata = result['metadata']
                context += f"Related Node {i+1} (Score: {result['score']:.4f}):\n"
                context += f"Source: {metadata.get('provenance', 'Unknown')}\n"
                
                if 'content' in metadata:
                    context += f"Content: {metadata['content']}\n"
                elif 'content_preview' in metadata:
                    context += f"Content Preview: {metadata['content_preview']}\n"
                else:
                    context += f"Base Document: {metadata.get('base_document', 'Unknown')}\n"
                
                # Find which entry point this is connected to
                connected_to = result.get('connected_to')
                if connected_to:
                    entry_index = next((j for j, ep in enumerate(entry_points) if ep['id'] == connected_to), None)
                    if entry_index is not None:
                        context += f"Connected to Entry Point: {entry_index + 1}\n"
                
                context += "\n"
            
            # Step 6: Create system prompt for graph-aware response
            system_prompt = """You are a helpful assistant that provides accurate information based on the 
            search results provided. Answer the query using only the information in the retrieved documents.
            If the information needed to answer the query is not in the documents, acknowledge this limitation.
            Be concise and specific."""
            
            # Step 7: Create the instruction for the LLM
            instruction = f"""
            Query: {query}
            
            Context:
            {context}
            
            Based on the above retrieved documents, please answer the query. 
            Provide specific details from the documents (which document, provenance etc) and indicate when information might be missing or uncertain.
            Do not give any score. Response should be in a conversational tone.
            Provide answer in elegant markdown format.
            """
            
            # Step 8: Get response from Groq
            answer = get_groq_response(instruction, system_prompt, model, temperature)
            
            # Return both results and answer
            return {
                'graph_results': all_results,
                'answer': answer
            }
    
    except Exception as e:
        error_message = f"Error in graph retrieval and answer generation: {str(e)}"
        st.error(error_message)
        return {
            'graph_results': [],
            'answer': f"An error occurred: {error_message}"
        }

def compare_search_approaches(query, semantic_result, graph_result, model="llama3-8b-8192", temperature=0.1):
    """Generate a comparison analysis between semantic search and graph-enhanced search results"""
    try:
        with st.spinner("Generating comparison analysis..."):
            system_prompt = """You are an analytical assistant that specializes in evaluating information retrieval 
            systems. Your task is to compare two different search approaches (semantic search vs. graph-enhanced search)
            and highlight the strengths and limitations of each approach for the given query."""
            
            # Extract answers from results
            semantic_answer = semantic_result.get('answer', 'No response available')
            graph_answer = graph_result.get('answer', 'No response available')
            
            instruction = f"""
            # Query
            {query}
            
            # Semantic Search Response
            {semantic_answer}
            
            # Graph-Enhanced Search Response
            {graph_answer}
            
            Please compare these two responses and analyze their differences, considering the following aspects:
            
            1. **Content Coverage**: Which response provides more comprehensive information? Does one miss important details that the other includes?
            
            2. **Context and Connections**: Does the graph-enhanced search reveal relationships or connections that semantic search misses? 
            
            3. **Factual Accuracy**: Are there any contradictions between the responses? If so, which one appears more accurate based on the information provided?
            
            4. **Relevance**: Which response better addresses the original query?
            
            5. **Unique Insights**: What unique information does each approach provide that the other doesn't?
            
            6. **Overall Assessment**: Which approach would be more useful for this particular query and why?
            
            Format your response in elegant markdown with clear headings and structure. Be specific about the differences, 
            citing examples from each response where relevant. Maintain a balanced analysis that recognizes the 
            strengths and limitations of each approach rather than simply declaring one "better" than the other.
            """
            
            # Get response from Groq
            comparison = get_groq_response(instruction, system_prompt, model, temperature)
            return comparison
    
    except Exception as e:
        error_message = f"Error generating comparison: {str(e)}"
        st.error(error_message)
        return f"An error occurred while comparing search approaches: {error_message}"

# App title
st.title("Semantic vs Graph Search Comparison")

# Sidebar with info and config
with st.sidebar:
    st.header("Configuration")
    
    model = st.selectbox(
        "LLM Model",
        ["llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
    )
    
    semantic_top_k = st.slider(
        "Semantic Search Results",
        min_value=3,
        max_value=20,
        value=10,
        help="Number of semantic search results to retrieve"
    )
    
    graph_neighbors_k = st.slider(
        "Graph Neighbors",
        min_value=2,
        max_value=15,
        value=5,
        help="Number of graph neighbors to retrieve for each semantic result"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controls randomness in the response generation"
    )
    
    comparison_temp = st.slider(
        "Comparison Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Temperature for comparison analysis generation"
    )
    
    st.markdown("---")
    st.markdown(
        """
        ### About
        This app compares two search approaches:
        - **Semantic Search**: Traditional search using semantic embeddings
        - **Graph Search**: Enhanced search using graph embeddings        
        """
    )

# Main query input
query = st.text_input("Enter your query:", placeholder="e.g., What was the relationship between Rome and Egypt?")

# Execute search button
if st.button("Execute Search"):
    if not query:
        st.warning("Please enter a query")
    else:
        # Store results in session state to access them for comparison
        st.session_state.semantic_result = semantic_retrieval_and_answer(
            query, 
            top_k=semantic_top_k, 
            model=model, 
            temperature=temperature
        )
        
        st.session_state.graph_result = graph_retrieval_and_answer(
            query, 
            semantic_top_k=semantic_top_k, 
            graph_neighbors_k=graph_neighbors_k, 
            model=model, 
            temperature=temperature
        )
        
        # Display results side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Semantic Search Results")
            st.markdown(st.session_state.semantic_result['answer'])
            
            # Optionally show raw results in an expander
            with st.expander("Show raw semantic results"):
                st.json(st.session_state.semantic_result['semantic_results'])
        
        with col2:
            st.header("Graph-Enhanced Search Results")
            st.markdown(st.session_state.graph_result['answer'])
            
            # Optionally show raw results in an expander
            with st.expander("Show raw graph results"):
                # Filter to just show entry points and their connections to keep it manageable
                entry_points = [r for r in st.session_state.graph_result['graph_results'] if r.get('is_entry_point', False)]
                st.json(entry_points)

# Comparison button - only shown if search has been executed
if 'semantic_result' in st.session_state and 'graph_result' in st.session_state:
    st.markdown("---")
    if st.button("Generate Comparison Analysis"):
        comparison = compare_search_approaches(
            query, 
            st.session_state.semantic_result, 
            st.session_state.graph_result, 
            model=model, 
            temperature=comparison_temp
        )
        
        st.header("Comparison Analysis")
        st.markdown(comparison)
        
        # Option to download the comparison
        st.download_button(
            label="Download Comparison Report",
            data=comparison,
            file_name=f"search_comparison_{int(time.time())}.md",
            mime="text/markdown"
        )