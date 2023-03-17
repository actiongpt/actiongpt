import os
import sys
import openai

def GPT_Completion(texts):
## Call the API key under your account (in a secure way)
    openai.api_key = "API-KEY"
    response = openai.Completion.create(
    engine="text-davinci-002",
    prompt =  texts,
    temperature = 0.5,
    top_p = 1,
    max_tokens = 128,
    frequency_penalty = 0,
    presence_penalty = 0
    )
    gpt3_texts = []
    for i in range(len(response.choices)):
        gpt3_texts.append(response.choices[i].text.replace("\n",""))
    
    return gpt3_texts