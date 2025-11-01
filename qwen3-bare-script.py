from llama_cpp import Llama

# Initialize model
llm = Llama(
    model_path=r"E:\ai\models\qwen3-8b-gguf\qwen3-8b-q5_k_m.gguf",
    n_ctx=8192,
    n_gpu_layers=-1,
    n_batch=512,
    n_threads=16,
    use_mmap=True,
    use_mlock=False,
    verbose=False,
    chat_format="chatml",
    main_gpu=0
)

# Generate function
def generate(prompt, **kwargs):
    params = {'max_tokens': 2048, 'temperature': 0.7, 'top_p': 0.95, 'top_k': 40, 'repeat_penalty': 1.1, 'stream': False}
    params.update(kwargs)
    return llm(prompt, **params)['choices'][0]['text']

# Stream generate function  
def stream_generate(prompt, **kwargs):
    params = {'max_tokens': 2048, 'temperature': 0.7, 'top_p': 0.95, 'top_k': 40, 'repeat_penalty': 1.1, 'stream': True}
    params.update(kwargs)
    output = ""
    for chunk in llm(prompt, **params):
        text = chunk['choices'][0]['text']
        print(text, end='', flush=True)
        output += text
    return output

# Chat function
def chat(messages, stream=True, **kwargs):
    params = {'max_tokens': 2048, 'temperature': 0.7, 'top_p': 0.95, 'stream': stream}
    params.update(kwargs)
    response = llm.create_chat_completion(messages=messages, **params)
    if stream:
        output = ""
        for chunk in response:
            if 'content' in chunk['choices'][0]['delta']:
                text = chunk['choices'][0]['delta']['content']
                print(text, end='', flush=True)
                output += text
        return output
    return response['choices'][0]['message']['content']

# Chat loop
def chat_loop():
    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'clear':
            messages = [messages[0]]
            continue
        elif not user_input:
            continue
        messages.append({"role": "user", "content": user_input})
        print("\nAssistant: ", end='')
        response = chat(messages)
        print()
        messages.append({"role": "assistant", "content": response})
        if len(messages) > 21:
            messages = [messages[0]] + messages[-20:]

if __name__ == "__main__":
    chat_loop()
