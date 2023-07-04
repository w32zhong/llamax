import fire
import torch
import gradio as gr
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

def main(model_path):
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    print('Loading weights')
    model = LlamaForCausalLM.from_pretrained(model_path)
    model.to('cuda')
    model.eval()
    device = model.device

    def inference(prompt='My name is Mariama, my favorite'):
        print('prompt:', prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                max_new_tokens=128
            )
        output = tokenizer.decode(generation_output[0])
        print('output:', output)
        return output

    inference()
    #iface = gr.Interface(fn=inference, inputs="text", outputs="text")
    #iface.launch(share=True)

if __name__ == '__main__':
    fire.Fire(main)
