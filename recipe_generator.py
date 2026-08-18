import time

import torch
import streamlit as st

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


RECIPE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@st.cache_resource
def load_recipe_model():

    tokenizer = AutoTokenizer.from_pretrained(
        RECIPE_MODEL
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        # Use all available CPU cores for inference
        torch.set_num_threads(torch.get_num_threads())

    model = AutoModelForCausalLM.from_pretrained(
        RECIPE_MODEL,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        low_cpu_mem_usage=True
    )

    model.eval()

    return tokenizer, model, device


def generate_recipe(
    food_name,
    servings=2,
    dietary_preference="Regular"
):

    load_start = time.time()

    tokenizer, recipe_model, device = load_recipe_model()

    load_time = time.time() - load_start

    prompt = f"""
You are generating a recipe for a food recognition application.

Detected food:
{food_name}

Required servings:
{servings}

Dietary preference:
{dietary_preference}

Follow these rules strictly:

1. The recipe MUST be specifically for {food_name}.
2. Do not change {food_name} into another dish.
3. Use ingredients that are appropriate and commonly used for {food_name}.
4. Do not add unrelated ingredients.
5. Make ingredient quantities appropriate for exactly {servings} servings.
6. Follow the requested dietary preference.
7. Use realistic cooking methods, temperatures, and cooking times.
8. Give practical instructions suitable for home cooking.
9. Do not mention any other serving size.
10. Do not include explanations, comments, or notes outside the recipe.
11. Keep the recipe concise.
12. If {food_name} requires dough, filling, sauce, marinade, or another component,
    include the necessary preparation steps.

Return ONLY the recipe using exactly this format:

## {food_name.title()} Recipe

### Ingredients
- Ingredient — quantity
- Ingredient — quantity
- Ingredient — quantity

### Instructions
1. Step one.
2. Step two.
3. Step three.
4. Step four.
5. Step five.

### Cooking Time
Preparation: XX minutes
Cooking: XX minutes
Total: XX minutes

### Servings
{servings}
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional chef and recipe assistant. "
                "Generate accurate, practical recipes and follow the "
                "user's requested food and serving size exactly."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(device)

    gen_start = time.time()

    with torch.inference_mode():

        outputs = recipe_model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    gen_time = time.time() - gen_start

    generated_tokens = outputs[
        0
    ][inputs["input_ids"].shape[1]:]

    recipe = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    # Timing info to help diagnose slow generation.
    # (model load is cached after the first run; only
    # generation time should matter on repeat calls)
    st.caption(
        f"Device: {device} | "
        f"Model load: {load_time:.2f}s | "
        f"Generation: {gen_time:.2f}s | "
        f"Tokens: {len(generated_tokens)}"
    )

    return recipe