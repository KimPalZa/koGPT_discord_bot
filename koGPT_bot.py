import discord
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("skt/kogpt2-base-v2")
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

client = discord.Client()

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    input_text = message.content
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    sample_output = model.generate(
        input_ids,
        do_sample=True, 
        max_length=100, 
        top_k=50, 
        top_p=0.95, 
        num_return_sequences=1
    )

    response_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    await message.channel.send(response_text)

client.run("YOUR_DISCORD_BOT_TOKEN")
