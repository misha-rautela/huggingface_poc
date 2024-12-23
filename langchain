#langchain and hugging face partner libraries
pip install langchain_huggingface
pip install huggingface_hub
pip install transformers
pip install accelerate
pip install bitsandbytes
pip install langchain
from langchain_huggingface import HuggingFaceEndpoint
import os
HF_TOKEN =''
HUGGINGFACEHUB_API_TOKEN =
os.environ["HUGGINGFACEHUB_API_TOKEN"]=''
repo_id="mistralai/Mistral-7B-Instruct-v0.3"
# simple LLM Modle endpoint integration
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token='')
llm.invoke("Misha Rautela apple Data Engineering")

#Langchain prompt template integration
from langchain import PromptTemplate, LLMChain

question="Who won the 100m in London Olympics"
template=""" Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
print(prompt)
llm_chain=LLMChain(llm=llm,prompt=prompt) 
print(llm_chain.run(question))

#Langchain transformer integration
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline

model_id="gpt2"
model=AutoModelForCausalLM.from_pretrained(model_id)
tokenizer=AutoTokenizer.from_pretrained(model_id)
pipe=pipeline("text-generation",model=model, tokenizer=tokenizer, max_new_tokens=100)
hf=HuggingFacePipeline(pipeline=pipe)
hf.invoke("Hugging face is a company ")

#Use of HuggingfacePipelines with GPU
gpu_llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    device=-1,
    pipeline_kwargs={"max_new_tokens":100},
)

from langchain import PromptTemplate
template=""" Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
chain=prompt|gpu_llm
question="Does misha rautela works as Engineer at Apple ?"
chain.invoke({"question":question})

