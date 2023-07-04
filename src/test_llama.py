import fire
import gradio as gr
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

def main(model_path):
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    print('Loading weights')
    model = LlamaForCausalLM.from_pretrained(model_path)#, torch_dtype=torch.float16)
    device = model.device
    print(device)
    #model.half()  # seems to fix bugs for some users.

    def inference(prompt='hello!'):
        print('prompt:', prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig()
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        print('output:', output)
        return output

    iface = gr.Interface(fn=inference, inputs="text", outputs="text")
    iface.launch(share=True)

if __name__ == '__main__':
    fire.Fire(main)
