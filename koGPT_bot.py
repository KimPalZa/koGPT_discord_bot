import torch
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
import discord
from discord.ext import commands

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>') 
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())
TOKEN = '*'

@bot.event
async def on_ready():
    print(f'Login bot: {bot.user}')

@bot.command()
async def 입력(message,*,vars):
    input_data = tokenizer.encode(vars)
    print(vars)
    gen = model.generate(torch.tensor([input_data]),max_length=128,
                             repetition_penalty=2.0,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id,
                             bos_token_id=tokenizer.bos_token_id,
                             use_cache=True)
    generated = tokenizer.decode(gen[0,:].tolist())
    print(generated)
    await message.channel.send(generated)

bot.run(TOKEN)
