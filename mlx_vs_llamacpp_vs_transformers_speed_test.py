import time
from typing import List, Dict
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from mlx_lm import load, generate

def test_llama_cpp(
    model_repo: str,
    model_filename: str,
    test_prompts: List[str],
    system_prompt: str = "",
    n_ctx: int = 2048
) -> Dict[str, float]:
    # Initialize llama.cpp model
    model = Llama.from_pretrained(
        repo_id=model_repo,
        filename=model_filename,
        n_ctx=n_ctx,
        verbose=False
    )
    
    # Warm-up
    _ = model.create_chat_completion(
        messages=[{"role": "user", "content": "warm up"}],
        max_tokens=10
    )
    
    # Testing
    times = []
    output_lens = []
    for prompt in test_prompts:
        start_time = time.time()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        output = model.create_chat_completion(
            messages=messages,
            max_tokens=50,
            temperature=0.0,
            top_p=1.0,
            repeat_penalty=1.0
        )
        output_len = len(output['choices'][0]['message']['content'])
        output_lens.append(output_len)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        "avg_time": sum(times) / len(times),
        "total_time": sum(times),
        "num_prompts": len(times),
        "output_lens": output_lens
    }

def test_transformers(
    model_name: str,
    test_prompts: List[str],
    system_prompt: str = "",
) -> Dict[str, float]:
    # Initialize transformers model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Warm-up
    conversation = tokenizer.apply_chat_template([{"role": "user", "content": "warm up"}], tokenize=True, return_tensors="pt").to(model.device)
    _ = model.generate(conversation, max_new_tokens=10)
    
    # Testing
    times = []
    output_lens = []
    for prompt in test_prompts:
        start_time = time.time()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        conversation = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to(model.device)
        output = model.generate(conversation, max_new_tokens=50, temperature=0.0, top_p=1.0, repetition_penalty=1.0)
        short_output = output[0, len(conversation[0]):]
        output_text = tokenizer.batch_decode([short_output], skip_special_tokens=True)
        output_len = len(output_text[0])
        output_lens.append(output_len)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        "avg_time": sum(times) / len(times),
        "total_time": sum(times),
        "num_prompts": len(times),
        "output_lens": output_lens
    }

def test_mlx(
    model_name: str,
    test_prompts: List[str],
    system_prompt: str = "",
) -> Dict[str, float]:
    """Test MLX-LM model performance."""
    # Initialize mlx model
    model, tokenizer = load(model_name)
    
    # Warm-up
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "warm up"}], tokenize=False)
    _ = generate(model, tokenizer, prompt, max_tokens=10)
    
    # Testing
    times = []
    output_lens = []
    for prompt in test_prompts:
        start_time = time.time()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Convert messages to MLX format
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        
        output = generate(
            model, 
            tokenizer, 
            formatted_prompt,
            max_tokens=50
        )
        output_len = len(output)
        output_lens.append(output_len)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        "avg_time": sum(times) / len(times),
        "total_time": sum(times),
        "num_prompts": len(times),
        "output_lens": output_lens
    }

def visualize_results():
    """
    Load the saved results and create beautiful visualizations using seaborn.
    Only shows average processing time and average output length.
    """
    # Load the results
    df = pd.read_csv('model_performance_results.csv')
    
    # Set the style
    sns.set_style("whitegrid")
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
    
    # Plot 1: Average Time Comparison
    sns.barplot(
        data=df,
        x='System Prompt',
        y='Avg Time (s)',
        hue='Model',
        ax=axes[0]
    )
    axes[0].set_title('Average Processing Time by Model and System Prompt')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Average Output Length
    sns.barplot(
        data=df,
        x='System Prompt',
        y='Avg Output Length',
        hue='Model',
        ax=axes[1]
    )
    axes[1].set_title('Average Output Length by Model and System Prompt')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('model_performance_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
        # Check if results already exist
    if os.path.exists('model_performance_results.csv'):
        print("Loading existing results...")
        visualize_results()
    else:
        # Test configuration
        TEST_PROMPTS = [
        "What is the capital of France?",
        "Explain quantum computing briefly.",
        "Write a hello world program in Python.",
        "What is the difference between a stack and a queue?",
        "How does a binary search tree work?",
        "Explain the concept of recursion in programming.",
        "What is the purpose of a hash table?",
        "How do you implement a sorting algorithm?",
        "What are the Seven Wonders of the World?",
        "What is the history of the Roman Empire?",
        "What is the significance of the Renaissance?",
        "What are the major religions of the world?",
        "What is the history of the United Nations?",
        "What are the major cultural festivals around the world?",
        "What is the history of the Olympic Games?",
        "What are the major historical events of the 20th century?",
        "How are you doing?",
        "What's your favorite hobby?",
        "Do you have any pets?",
        "What's your favorite food?",
    ]
    
        SHORT_SYSTEM_PROMPT = "You are a helpful AI assistant."
        LONG_SYSTEM_PROMPT = """You are a helpful AI assistant with expertise in various fields including science, technology, programming, and general knowledge. 
        You should provide accurate, concise, and well-structured responses. When discussing technical topics, use clear explanations suitable for the user's level of understanding. 
        For programming questions, include relevant code examples and explanations. Always maintain a professional and friendly tone while being direct and efficient in your communication."""
        
        EXTRA_LONG_SYSTEM_PROMPT = """You are an advanced AI assistant with comprehensive expertise across multiple disciplines including computer science, mathematics, physics, biology, engineering, technology, programming, and general knowledge. Your responses should be tailored to provide maximum value while maintaining clarity and precision.

        When addressing technical topics, you should:
        1. Begin with a high-level overview suitable for beginners
        2. Progressively introduce more complex concepts
        3. Include relevant real-world examples and applications
        4. Provide code samples when appropriate, with detailed comments
        5. Highlight potential pitfalls and best practices

        Your communication style should be:
        - Professional yet approachable
        - Clear and well-structured
        - Precise and accurate
        - Engaging and informative
        - Adaptable to the user's expertise level

        For programming-related queries:
        - Include working code examples with proper formatting
        - Explain the underlying concepts and logic
        - Discuss time and space complexity when relevant
        - Suggest alternative approaches and optimizations
        - Address potential edge cases and error handling

        Remember to maintain a balance between thoroughness and conciseness, always focusing on delivering practical value while ensuring comprehensive understanding of the subject matter."""
        
        # Test with no system prompt
        print("\nTesting without system prompt:")
        llama_results = test_llama_cpp(
            model_repo="andito/SmolLM2-1.7B-Intermediate-SFT-v2-summarization-lora-r32-a64-merged-2-F16-GGUF",
            model_filename="smollm2-1.7b-intermediate-sft-v2-summarization-lora-r32-a64-merged-2-f16.gguf",
            test_prompts=TEST_PROMPTS
        )
        
        transformers_results = test_transformers(
            model_name="HuggingFaceTB/SmolLM2-1.7B-Intermediate-SFT-v2-summarization-lora-r32-a64-merged-2",
            test_prompts=TEST_PROMPTS
        )
        
        mlx_results = test_mlx(
            model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            test_prompts=TEST_PROMPTS
        )
        
        # Test with short system prompt
        print("\nTesting with short system prompt:")
        llama_results_short = test_llama_cpp(
            model_repo="andito/SmolLM2-1.7B-Intermediate-SFT-v2-summarization-lora-r32-a64-merged-2-F16-GGUF",
            model_filename="smollm2-1.7b-intermediate-sft-v2-summarization-lora-r32-a64-merged-2-f16.gguf",
            test_prompts=TEST_PROMPTS,
            system_prompt=SHORT_SYSTEM_PROMPT
        )
        
        transformers_results_short = test_transformers(
            model_name="HuggingFaceTB/SmolLM2-1.7B-Intermediate-SFT-v2-summarization-lora-r32-a64-merged-2",
            test_prompts=TEST_PROMPTS,
            system_prompt=SHORT_SYSTEM_PROMPT
        )
        
        mlx_results_short = test_mlx(
            model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            test_prompts=TEST_PROMPTS,
            system_prompt=SHORT_SYSTEM_PROMPT
        )
        
        # Test with long system prompt
        print("\nTesting with long system prompt:")
        llama_results_long = test_llama_cpp(
            model_repo="andito/SmolLM2-1.7B-Intermediate-SFT-v2-summarization-lora-r32-a64-merged-2-F16-GGUF",
            model_filename="smollm2-1.7b-intermediate-sft-v2-summarization-lora-r32-a64-merged-2-f16.gguf",
            test_prompts=TEST_PROMPTS,
            system_prompt=LONG_SYSTEM_PROMPT
        )
        
        transformers_results_long = test_transformers(
            model_name="HuggingFaceTB/SmolLM2-1.7B-Intermediate-SFT-v2-summarization-lora-r32-a64-merged-2",
            test_prompts=TEST_PROMPTS,
            system_prompt=LONG_SYSTEM_PROMPT
        )
        
        mlx_results_long = test_mlx(
            model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            test_prompts=TEST_PROMPTS,
            system_prompt=LONG_SYSTEM_PROMPT
        )
        
        # Test with extra long system prompt
        print("\nTesting with extra long system prompt:")
        llama_results_extra_long = test_llama_cpp(
            model_repo="andito/SmolLM2-1.7B-Intermediate-SFT-v2-summarization-lora-r32-a64-merged-2-F16-GGUF",
            model_filename="smollm2-1.7b-intermediate-sft-v2-summarization-lora-r32-a64-merged-2-f16.gguf",
            test_prompts=TEST_PROMPTS,
            system_prompt=EXTRA_LONG_SYSTEM_PROMPT
        )
        
        transformers_results_extra_long = test_transformers(
            model_name="HuggingFaceTB/SmolLM2-1.7B-Intermediate-SFT-v2-summarization-lora-r32-a64-merged-2",
            test_prompts=TEST_PROMPTS,
            system_prompt=EXTRA_LONG_SYSTEM_PROMPT
        )
        
        mlx_results_extra_long = test_mlx(
            model_name="mlx-community/SmolLM2-1.7B-Instruct",
            test_prompts=TEST_PROMPTS,
            system_prompt=EXTRA_LONG_SYSTEM_PROMPT
        )
        
        # Create DataFrame with results
        results_data = []
        for test_name, results in [
            ("No System Prompt - Llama.cpp", llama_results),
            ("No System Prompt - Transformers", transformers_results),
            ("No System Prompt - MLX", mlx_results),
            ("Short System Prompt - Llama.cpp", llama_results_short),
            ("Short System Prompt - Transformers", transformers_results_short),
            ("Short System Prompt - MLX", mlx_results_short),
            ("Long System Prompt - Llama.cpp", llama_results_long),
            ("Long System Prompt - Transformers", transformers_results_long),
            ("Long System Prompt - MLX", mlx_results_long),
            ("Extra Long System Prompt - Llama.cpp", llama_results_extra_long),
            ("Extra Long System Prompt - Transformers", transformers_results_extra_long),
            ("Extra Long System Prompt - MLX", mlx_results_extra_long),
        ]:
            system_prompt_type = test_name.split(" - ")[0]
            model_type = test_name.split(" - ")[1]
            
            results_data.append({
                'System Prompt': system_prompt_type,
                'Model': model_type,
                'Avg Time (s)': round(results['avg_time'], 3),
                'Total Time (s)': round(results['total_time'], 3),
                'Num Prompts': results['num_prompts'],
                'Avg Output Length': round(sum(results['output_lens']) / len(results['output_lens']), 1),
                'Min Output Length': min(results['output_lens']),
                'Max Output Length': max(results['output_lens'])
            })
        
        df = pd.DataFrame(results_data)
        
        # Save results to CSV
        df.to_csv('model_performance_results.csv', index=False)
        
        # Print formatted results
        print("\nDetailed Performance Comparison:")
        print("=" * 100)
        
        # Print summary grouped by system prompt type
        for prompt_type in ['No System Prompt', 'Short System Prompt', 'Long System Prompt', 'Extra Long System Prompt']:
            print(f"\n{prompt_type}:")
            print("-" * 100)
            subset = df[df['System Prompt'] == prompt_type]
            print(subset.to_string(index=False))
        
        # Print overall statistics
        print("\nSummary by Model Type:")
        print("-" * 100)
        summary = df.groupby('Model').agg({
            'Avg Time (s)': 'mean',
            'Total Time (s)': 'sum',
            'Avg Output Length': 'mean'
        }).round(3)
        print(summary.to_string())
        
        # After all tests are complete and results are saved
        visualize_results()