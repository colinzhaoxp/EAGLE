import torch

from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template

try:
    import debugpy
    debugpy.listen(("localhost", 9501))
    print("watiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    print(e)

base_model_path = "./pretrain/vicuna-7b-v1.3"
EAGLE_model_path = "./pretrain/EAGLE-Vicuna-7B-v1.3"


model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-1
)
model.eval()

your_message="Hello"

conv = get_conversation_template("vicuna")

conv.append_message(conv.roles[0], your_message)
conv.append_message(conv.roles[1], None)

# prompt = conv.get_prompt()
prompt = your_message
input_ids=model.tokenizer([prompt]).input_ids

input_ids = torch.as_tensor(input_ids).cuda()

output_ids=model.eagenerate(input_ids,temperature=0.5,max_new_tokens=512)
output=model.tokenizer.decode(output_ids[0])

print(output)