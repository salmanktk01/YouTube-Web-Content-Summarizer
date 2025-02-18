import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders import UnstructuredURLLoader

from langchain_huggingface import HuggingFaceEndpoint
import pdb
## sstreamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')


## Get the Groq API Key and url(YT or website)to be summarized
with st.sidebar:
    hf_api_key=st.text_input("Hugging face API",value="",type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")

## Gemma Model USsing Groq API
# llm =ChatGroq(model="Llama3-8b-8192", groq_api_key="gsk_U8W5VhMUx4v92NzX9YiWGdyb3FY0jnB2KMwh34F2P84KzU")
repo_id="mistralai/Mistral-Nemo-Instruct-2407" #the model is text generation model 
llm=HuggingFaceEndpoint(repo_id=repo_id,
    max_new_tokens=150,  # directly pass as a parameter
    temperature=0.1,     # directly pass as a parameter
    huggingfacehub_api_token="hf_yaCsdNdiVvDvnIxamliomAYSBdO")

prompt_template="""
Provide a summary of the following content in 400 words:
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if  not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT video URL or website URL")

    else:
        try:
            with st.spinner("Waiting..."):
            ## loading the website or YT video data
             if "youtube.com" in generic_url:
                try:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                except Exception as e:
                    st.error(f"Failed to load YouTube video: {e}")
                    raise e
             else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": (
                                "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/116.0.0.0 Safari/537.36"
                            )
                        }
                    )
                  
                
         
            docs = loader.load()
            # pdb=pdb.set_trace() 
            # Chain For Summarization
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
            output_summary = chain.run(docs)
            
            st.success("Summary generated successfully!")
            st.write(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")






#pytube error innertube mein jao and client web kr dena 
