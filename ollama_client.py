import requests
import csv
from typing import List, Tuple

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"
REPORT_FILENAME = "inference_report.csv"


def query_ollama(prompt: str, model: str = MODEL_NAME, url: str = OLLAMA_URL) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "Ответ пустой")
    except requests.exceptions.RequestException as e:
        print(f"Не удалось получить ответ от Ollama: {e}")
        return f"{str(e)}"


def run_inference_and_save_report(prompts: List[str], filename: str = REPORT_FILENAME) -> None:
    results: List[Tuple[str, str]] = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Обрабатываю: {prompt[:50]}...")
        response = query_ollama(prompt)
        results.append((prompt, response))

    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["запрос", "ответ"])
        writer.writerows(results)


if __name__ == "__main__":
    test_prompts = [
        "Что такое машинное обучение?",
        "Напиши короткий стих о весеннем дожде.",
        "Объясни разницу между фотосинтезом и дыханием растений.",
        "Какие основные принципы ООП ты знаешь?",
        "Переведи на английский: 'Киберфизические системы'",
        "Почему небо голубое? Объясни простыми словами.",
        "Напиши рецепт простого завтрака из 5 ингредиентов.",
        "Что такое блокчейн и где он применяется?",
        "Перечисли 5 интересных фактов о космосе.",
        "Объясни, как работает микроволновая печь."
    ]
    run_inference_and_save_report(test_prompts)